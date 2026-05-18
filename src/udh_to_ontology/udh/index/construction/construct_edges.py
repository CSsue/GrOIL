from src.dispergraph_to_ontology.dispergraph.models.Graph import Graph
from src.dispergraph_to_ontology.dispergraph.models.index_objects import *
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from src.dispergraph_to_ontology.dispergraph.index.extraction.extract_relations import *
from src.dispergraph_to_ontology.dispergraph.utils.general import batch_list

def construct_edges_between_chunks(graph: Graph):
    chunk_ids = graph.get_chunk_ids()
    chunk_edge_constructor = dspy.ChainOfThought(ChunkEdge)

    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            chunk1_id = chunk_ids[i]
            chunk2_id = chunk_ids[j]

            # check if there exists an edge between chunk 1 and chunk 2
            if graph.has_chunk_chunk_edge(chunk1_id, chunk2_id):
                continue

            # if there does not exist an edge between chunk 1 and chunk 2, then you can create an edge

            # get chunk summaries
            chunk1_summary = graph.chunks[chunk1_id].summary
            chunk2_summary = graph.chunks[chunk2_id].summary

            # llm call for edge info
            edge_info = chunk_edge_constructor(chunk_summary_1=chunk1_summary, chunk_summary_2=chunk2_summary)

            # add edges between chunks in both directions since this is directed graph with different descriptions
            graph.network.add_edge(chunk1_id, chunk2_id,
                                   description=edge_info.description_1_2,
                                   label=edge_info.label_1_2,
                                   edge_type="chunk_chunk",
                                   score=edge_info.relevance_score,
                                   fact_comparison=False)

            graph.network.add_edge(chunk2_id, chunk1_id,
                                   description=edge_info.description_2_1,
                                   label=edge_info.label_2_1,
                                   edge_type="chunk_chunk",
                                   score=edge_info.relevance_score,
                                   fact_comparison=False)

    print("Finished constructing edges between chunks")


# 1. Define a pure worker function that ONLY talks to the LLM and returns data
def _chunk_edge_worker(chunk_edge_constructor, c1_id, c1_summary, c2_id, c2_summary):
    # This is the slow part. It happens in parallel.
    edge_info = chunk_edge_constructor(chunk_summary_1=c1_summary, chunk_summary_2=c2_summary)

    # Return the IDs and the result. DO NOT modify the graph here.
    return c1_id, c2_id, edge_info


def construct_edges_between_chunks_parallel(graph: Graph, max_workers: int = 10):
    chunk_ids = graph.get_chunk_ids()
    chunk_edge_constructor = dspy.ChainOfThought(ChunkEdge)

    # 2. Gather all the tasks upfront
    tasks = []
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            chunk1_id = chunk_ids[i]
            chunk2_id = chunk_ids[j]

            # Filter out existing edges BEFORE sending to threads
            if not graph.has_chunk_chunk_edge(chunk1_id, chunk2_id):
                tasks.append((
                    chunk1_id, graph.chunks[chunk1_id].summary,
                    chunk2_id, graph.chunks[chunk2_id].summary
                ))

    # 3. Execute the slow LLM calls in parallel
    print(f"Executing {len(tasks)} chunk comparisons across {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        futures = {
            executor.submit(_chunk_edge_worker, chunk_edge_constructor, t[0], t[1], t[2], t[3]): t
            for t in tasks
        }

        # 4. Safely update the graph in the main thread as results finish
        for future in as_completed(futures):
            try:
                c1_id, c2_id, edge_info = future.result()

                # Because this loop is in the main thread, networkx mutations are safe
                graph.network.add_edge(c1_id, c2_id,
                                       description=edge_info.description_1_2,
                                       label=edge_info.label_1_2,
                                       score=edge_info.relevance_score,
                                       edge_type="chunk_chunk",
                                       fact_comparison=False)

                graph.network.add_edge(c2_id, c1_id,
                                       description=edge_info.description_2_1,
                                       label=edge_info.label_2_1,
                                       score=edge_info.relevance_score,
                                       edge_type="chunk_chunk",
                                       fact_comparison=False)
            except Exception as e:
                print(f"Error processing chunk edge: {e}")

    print("Finished constructing edges between chunks")

def construct_edges_between_same_chunk_facts(graph: Graph):
    fact_edge_constructor = dspy.ChainOfThought(SameChunkFactEdge)

    for chunk_id in graph.get_chunk_ids():
        fact_ids = graph.get_fact_ids(by_chunk_id=chunk_id)
        for i in range(len(fact_ids)):
            for j in range(i + 1, len(fact_ids)):
                fact_id1 = fact_ids[i]
                fact_id2 = fact_ids[j]

                # check if edge data already exists
                edge_data = graph.network.get_edge_data(fact_id1, fact_id2)
                if not edge_data is None:
                    continue

                # get info
                chunk_summary = graph.chunks[chunk_id].summary
                fact_sentence_1 = graph.facts[fact_id1].sentence
                fact_sentence_2 = graph.facts[fact_id2].sentence

                # call llm to construct edge info
                edge_info = fact_edge_constructor(chunk_summary=chunk_summary,
                                                  fact_sentence_1=fact_sentence_1,
                                                  fact_sentence_2=fact_sentence_2)

                # create the edges
                graph.network.add_edge(fact_id1, fact_id2,
                                       description=edge_info.description_1_2,
                                       label=edge_info.label_1_2,
                                       edge_type="fact_fact",
                                       score=edge_info.relevance_score)

                graph.network.add_edge(fact_id2, fact_id1,
                                       description=edge_info.description_2_1,
                                       label=edge_info.label_2_1,
                                       edge_type="fact_fact",
                                       score=edge_info.relevance_score)

    print("Finished constructing edges between facts in the same chunk")


def _same_chunk_fact_worker(fact_edge_constructor, chunk_summary, f1_id, f1_sentence, f2_id, f2_sentence):
    # This is the slow I/O bound part that happens in parallel
    edge_info = fact_edge_constructor(
        chunk_summary=chunk_summary,
        fact_sentence_1=f1_sentence,
        fact_sentence_2=f2_sentence
    )

    # Return the IDs and the generated info. DO NOT modify the graph here.
    return f1_id, f2_id, edge_info


def construct_edges_between_same_chunk_facts_parallel(graph: Graph, max_workers: int = 10):
    fact_edge_constructor = dspy.ChainOfThought(SameChunkFactEdge)

    # 2. Gather all the tasks upfront
    tasks = []

    for chunk_id in graph.get_chunk_ids():
        fact_ids = graph.get_fact_ids(by_chunk_id=chunk_id)
        chunk_summary = graph.chunks[chunk_id].summary

        for i in range(len(fact_ids)):
            for j in range(i + 1, len(fact_ids)):
                fact_id1 = fact_ids[i]
                fact_id2 = fact_ids[j]

                # Filter out existing edges BEFORE sending to threads
                if not graph.network.has_edge(fact_id1, fact_id2):
                    fact_sentence_1 = graph.facts[fact_id1].sentence
                    fact_sentence_2 = graph.facts[fact_id2].sentence

                    # Pack the required data into a tuple
                    tasks.append((
                        chunk_summary,
                        fact_id1, fact_sentence_1,
                        fact_id2, fact_sentence_2
                    ))

    # 3. Execute the slow LLM calls in parallel
    print(f"Executing {len(tasks)} intra-chunk fact comparisons across {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        futures = {
            executor.submit(
                _same_chunk_fact_worker,
                fact_edge_constructor,
                t[0], t[1], t[2], t[3], t[4]
            ): t
            for t in tasks
        }

        # 4. Safely update the graph in the main thread as results finish
        for future in as_completed(futures):
            try:
                f1_id, f2_id, edge_info = future.result()

                # Because this loop is in the main thread, networkx mutations are safe
                graph.network.add_edge(
                    f1_id, f2_id,
                    description=edge_info.description_1_2,
                    label=edge_info.label_1_2,
                    edge_type="fact_fact",
                    score=edge_info.relevance_score
                )

                graph.network.add_edge(
                    f2_id, f1_id,
                    description=edge_info.description_2_1,
                    label=edge_info.label_2_1,
                    edge_type="fact_fact",
                    score=edge_info.relevance_score
                )
            except Exception as e:
                print(f"Error processing intra-chunk fact edge: {e}")

    print("Finished constructing edges between facts in the same chunk")


def construct_edges_between_all_facts(graph: Graph, threshold: float = 0.5):
    """
    Evaluates highly related chunks and creates deep fact-to-fact relationships
    between them to build cross-chunk context within the knowledge graph.
    """
    # Create DSPy constructors
    fact_edge_constructor = dspy.ChainOfThought(FactEdge)
    candidate_selector = dspy.ChainOfThought(CandidateFacts)

    chunk_ids = graph.get_chunk_ids()

    # Pairwise comparison between all chunks
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            chunk1_id = chunk_ids[i]
            chunk2_id = chunk_ids[j]

            # Get edge data between the chunks
            edge_data_1_to_2 = graph.network.get_edge_data(chunk1_id, chunk2_id)
            edge_data_2_to_1 = graph.network.get_edge_data(chunk2_id, chunk1_id)

            # If there does not exist edge data between chunks, skip
            if edge_data_1_to_2 is None or edge_data_2_to_1 is None:
                continue

            # Check if we have already cross-examined facts between these chunks
            if edge_data_1_to_2.get("fact_comparison") is True:
                continue

            # If the edge between chunks has a score above the threshold
            score_1_to_2 = edge_data_1_to_2.get("score", 0.0)

            if score_1_to_2 >= threshold:
                # Extract chunk summaries
                chunk1_summary = graph.chunks[chunk1_id].summary
                chunk2_summary = graph.chunks[chunk2_id].summary

                # Get dictionary of fact IDs to sentences for each chunk
                fact_dict_1 = graph.get_fact_ids(by_chunk_id=chunk1_id, mode="dict")
                fact_dict_2 = graph.get_fact_ids(by_chunk_id=chunk2_id, mode="dict")

                # Extract rich edge descriptions from the chunk-to-chunk level
                desc_1_to_2 = edge_data_1_to_2.get("description", "")
                desc_2_to_1 = edge_data_2_to_1.get("description", "")

                # Select candidate facts via DSPy selector for Chunk 1
                cand_facts_1_result = candidate_selector(
                    chunk_summary_1=chunk1_summary,
                    chunk_summary_2=chunk2_summary,
                    rich_edge_description=desc_1_to_2,
                    fact_dict=fact_dict_1
                )

                # Select candidate facts via DSPy selector for Chunk 2
                # (Swapping summaries so it views the relationship from Chunk 2's perspective)
                cand_facts_2_result = candidate_selector(
                    chunk_summary_1=chunk2_summary,
                    chunk_summary_2=chunk1_summary,
                    rich_edge_description=desc_2_to_1,
                    fact_dict=fact_dict_2
                )

                # Filter the LLM output list for fact IDs that actually exist in the graph
                valid_cand_facts_1 = [fid for fid in cand_facts_1_result.candidate_facts if fid in graph.facts]
                valid_cand_facts_2 = [fid for fid in cand_facts_2_result.candidate_facts if fid in graph.facts]

                # Pairwise comparison between the candidate facts of chunk 1 and chunk 2
                for fact1_id in valid_cand_facts_1:
                    for fact2_id in valid_cand_facts_2:

                        # Between the candidate facts, check if there already exists an edge
                        if graph.network.has_edge(fact1_id, fact2_id):
                            continue

                        # Extract important info needed for the LLM call
                        fact1_sentence = graph.facts[fact1_id].sentence
                        fact2_sentence = graph.facts[fact2_id].sentence

                        # Call LLM via DSPy constructor to create edge info
                        fact_edge_info = fact_edge_constructor(
                            fact_sentence_1=fact1_sentence,
                            chunk_summary_1=chunk1_summary,
                            fact_sentence_2=fact2_sentence,
                            chunk_summary_2=chunk2_summary
                        )

                        # Create an edge both ways between each fact
                        graph.network.add_edge(
                            fact1_id, fact2_id,
                            description=fact_edge_info.description_1_2,
                            label=fact_edge_info.label_1_2,
                            edge_type="fact_fact",
                            score=fact_edge_info.relevance_score
                        )

                        graph.network.add_edge(
                            fact2_id, fact1_id,
                            description=fact_edge_info.description_2_1,
                            label=fact_edge_info.label_2_1,
                            edge_type="fact_fact",
                            score=fact_edge_info.relevance_score
                        )

                # Finally, set the attribute fact_comparison to True to prevent redundant future checks
                graph.network[chunk1_id][chunk2_id]["fact_comparison"] = True
                graph.network[chunk2_id][chunk1_id]["fact_comparison"] = True
    print("Finished constructing edges between all facts.")


def _candidate_worker(candidate_selector, c1_id, c2_id, c1_sum, c2_sum, desc_1_2, desc_2_1, dict_1, dict_2):
    """Worker for Phase 1: Retrieves candidate facts for both directions of a chunk pair."""
    cand_1_res = candidate_selector(
        chunk_summary_1=c1_sum,
        chunk_summary_2=c2_sum,
        rich_edge_description=desc_1_2,
        fact_dict=dict_1
    )

    cand_2_res = candidate_selector(
        chunk_summary_1=c2_sum,
        chunk_summary_2=c1_sum,
        rich_edge_description=desc_2_1,
        fact_dict=dict_2
    )

    return c1_id, c2_id, cand_1_res.candidate_facts, cand_2_res.candidate_facts


def _cross_chunk_fact_worker(fact_edge_constructor, f1_id, f2_id, f1_sent, f2_sent, c1_sum, c2_sum):
    """Worker for Phase 2: Generates the edge description between two candidate facts."""
    edge_info = fact_edge_constructor(
        fact_sentence_1=f1_sent,
        chunk_summary_1=c1_sum,
        fact_sentence_2=f2_sent,
        chunk_summary_2=c2_sum
    )
    return f1_id, f2_id, edge_info


# --- MAIN PARALLELIZED FUNCTION ---

def construct_edges_between_all_facts_parallel(graph: Graph, threshold: float = 0.5, max_workers: int = 10):
    """
    Evaluates highly related chunks and creates deep fact-to-fact relationships
    between them to build cross-chunk context within the knowledge graph.
    """
    fact_edge_constructor = dspy.ChainOfThought(FactEdge)
    candidate_selector = dspy.ChainOfThought(CandidateFacts)

    chunk_ids = graph.get_chunk_ids()
    candidate_tasks = []

    # ==========================================
    # PREPARATION: Gather chunk pairs for Phase 1
    # ==========================================
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            chunk1_id = chunk_ids[i]
            chunk2_id = chunk_ids[j]

            edge_data_1_to_2 = graph.network.get_edge_data(chunk1_id, chunk2_id)
            edge_data_2_to_1 = graph.network.get_edge_data(chunk2_id, chunk1_id)

            if edge_data_1_to_2 is None or edge_data_2_to_1 is None:
                continue
            if edge_data_1_to_2.get("fact_comparison") is True:
                continue

            score_1_to_2 = edge_data_1_to_2.get("score", 0.0)
            if score_1_to_2 >= threshold:
                candidate_tasks.append((
                    chunk1_id, chunk2_id,
                    graph.chunks[chunk1_id].summary,
                    graph.chunks[chunk2_id].summary,
                    edge_data_1_to_2.get("description", ""),
                    edge_data_2_to_1.get("description", ""),
                    graph.get_fact_ids(by_chunk_id=chunk1_id, mode="dict"),
                    graph.get_fact_ids(by_chunk_id=chunk2_id, mode="dict")
                ))

    # ==========================================
    # PHASE 1: Parallel Candidate Fact Selection
    # ==========================================
    candidate_results = []
    print(f"Phase 1: Executing {len(candidate_tasks)} Candidate Fact selections across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_candidate_worker, candidate_selector, *t): t
            for t in candidate_tasks
        }
        for future in as_completed(futures):
            try:
                candidate_results.append(future.result())
            except Exception as e:
                print(f"Error processing candidate selector: {e}")

    # ==========================================
    # PREPARATION: Gather fact pairs for Phase 2
    # ==========================================
    fact_edge_tasks = []
    chunks_to_mark = []  # We track this to mark them as completed later

    for c1_id, c2_id, cands_1, cands_2 in candidate_results:
        # Save these chunk IDs so we can flag them as completed at the end
        chunks_to_mark.append((c1_id, c2_id))

        # Filter valid facts directly from the LLM outputs
        valid_1 = [fid for fid in cands_1 if fid in graph.facts]
        valid_2 = [fid for fid in cands_2 if fid in graph.facts]

        c1_summary = graph.chunks[c1_id].summary
        c2_summary = graph.chunks[c2_id].summary

        for f1_id in valid_1:
            for f2_id in valid_2:
                # Filter out existing edges before doing LLM math
                if not graph.network.has_edge(f1_id, f2_id):
                    fact_edge_tasks.append((
                        f1_id, f2_id,
                        graph.facts[f1_id].sentence,
                        graph.facts[f2_id].sentence,
                        c1_summary, c2_summary
                    ))

    # ==========================================
    # PHASE 2: Parallel Cross-Chunk Fact Linking
    # ==========================================
    print(f"Phase 2: Executing {len(fact_edge_tasks)} Cross-Chunk Fact evaluations across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_cross_chunk_fact_worker, fact_edge_constructor, *t): t
            for t in fact_edge_tasks
        }
        for future in as_completed(futures):
            try:
                f1_id, f2_id, edge_info = future.result()

                # Safely write to NetworkX in the main thread
                graph.network.add_edge(
                    f1_id, f2_id,
                    description=edge_info.description_1_2,
                    label=edge_info.label_1_2,
                    edge_type="fact_fact",
                    score=edge_info.relevance_score
                )
                graph.network.add_edge(
                    f2_id, f1_id,
                    description=edge_info.description_2_1,
                    label=edge_info.label_2_1,
                    edge_type="fact_fact",
                    score=edge_info.relevance_score
                )
            except Exception as e:
                print(f"Error processing cross-chunk fact edge: {e}")

    # ==========================================
    # CLEANUP: Mark chunk comparisons as complete
    # ==========================================
    for c1_id, c2_id in chunks_to_mark:
        graph.network[c1_id][c2_id]["fact_comparison"] = True
        graph.network[c2_id][c1_id]["fact_comparison"] = True

    print("Finished constructing edges between all facts.")

def construct_edges_between_entities_and_facts(graph: Graph, entity_batch_size=5):
    """
    Constructs edges going from Entities back to their parent Facts.
    Uses batched LLM calls to generate the specific relationship labels.
    """
    # 1. Create DSPy constructor
    entity_fact_edge_constructor = dspy.ChainOfThought(EntityFactEdge)

    # 2. Collect all fact IDs
    fact_ids = graph.get_fact_ids()

    for fact_id in fact_ids:
        # 3. Get name and sentence for this fact
        fact = graph.facts[fact_id]
        fact_name = fact.name
        fact_sentence = fact.sentence

        # 4. Find the set of entities that the fact is connected to
        connected_entity_ids = []
        for _, target_node, edge_data in graph.network.out_edges(fact_id, data=True):
            if edge_data.get("edge_type") == "fact_entity":
                connected_entity_ids.append(target_node)

        if not connected_entity_ids:
            continue

        # 5. Batch the entities using the general utils function
        entity_batches = batch_list(connected_entity_ids, max_batch_size=entity_batch_size)

        for batch in entity_batches:
            filtered_entities = []
            name_to_id_map = {}

            # 6. Filter entities by checking if the reverse edge (Entity -> Fact) already exists
            for ent_id in batch:
                if not graph.network.has_edge(ent_id, fact_id):
                    filtered_entities.append(ent_id)
                    # Keep dictionary mapping entity name to entity ID for the LLM
                    ent_name = graph.entities[ent_id].name
                    name_to_id_map[ent_name] = ent_id

            # Skip if all entities in this batch already have reverse edges
            if not filtered_entities:
                continue

            # 7. Call LLM to create edge info
            entity_names_list = list(name_to_id_map.keys())

            try:
                edge_info = entity_fact_edge_constructor(
                    fact_name=fact_name,
                    fact_sentence=fact_sentence,
                    EntityNames=entity_names_list
                )

                # 8. Create the edges using the mapped IDs
                for relation in edge_info.relations:
                    generated_name = relation.entity_name
                    relation_label = relation.relation_name

                    # Ensure the LLM didn't hallucinate a slightly different name
                    if generated_name in name_to_id_map:
                        mapped_ent_id = name_to_id_map[generated_name]

                        graph.network.add_edge(
                            mapped_ent_id,
                            fact_id,
                            edge_type="entity_fact",
                            label=relation_label
                        )
            except Exception as e:
                # Always good to catch exceptions per batch so one bad LLM output doesn't crash the loop
                print(f"Error generating entity-fact edges for fact {fact_name}: {e}")

    print("Finished constructing edges between entities and facts.")


def _entity_fact_worker(entity_fact_edge_constructor, fact_id, fact_name, fact_sentence, entity_names_list,
                        name_to_id_map):
    """
    Worker for isolating the LLM call.
    It passes through the fact_id and the name_to_id_map so the main thread
    has everything it needs to safely reconstruct the graph edges.
    """
    edge_info = entity_fact_edge_constructor(
        fact_name=fact_name,
        fact_sentence=fact_sentence,
        EntityNames=entity_names_list
    )

    # Return the mapped data alongside the DSPy output
    return fact_id, name_to_id_map, edge_info


# --- MAIN PARALLELIZED FUNCTION ---

def construct_edges_between_entities_and_facts_parallel(graph: Graph, entity_batch_size: int = 5, max_workers: int = 10):
    """
    Constructs edges going from Entities back to their parent Facts.
    Uses threaded, batched LLM calls to generate the specific relationship labels.
    """
    entity_fact_edge_constructor = dspy.ChainOfThought(EntityFactEdge)
    fact_ids = graph.get_fact_ids()

    tasks = []

    # ==========================================
    # PREPARATION: Gather tasks
    # ==========================================
    for fact_id in fact_ids:
        fact = graph.facts[fact_id]
        fact_name = fact.name
        fact_sentence = fact.sentence

        # Find the set of entities that the fact is connected to
        connected_entity_ids = []
        for _, target_node, edge_data in graph.network.out_edges(fact_id, data=True):
            if edge_data.get("edge_type") == "fact_entity":
                connected_entity_ids.append(target_node)

        if not connected_entity_ids:
            continue

        # Batch the entities
        entity_batches = batch_list(connected_entity_ids, max_batch_size=entity_batch_size)

        for batch in entity_batches:
            name_to_id_map = {}

            # Filter entities by checking if the reverse edge already exists
            for ent_id in batch:
                if not graph.network.has_edge(ent_id, fact_id):
                    ent_name = graph.entities[ent_id].name
                    name_to_id_map[ent_name] = ent_id

            # If all entities in this batch are already linked, skip
            if not name_to_id_map:
                continue

            # Add to the task list
            entity_names_list = list(name_to_id_map.keys())
            tasks.append((
                fact_id,
                fact_name,
                fact_sentence,
                entity_names_list,
                name_to_id_map
            ))

    # ==========================================
    # EXECUTION: Parallel LLM Calls
    # ==========================================
    print(f"Executing {len(tasks)} Entity-Fact batches across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_entity_fact_worker, entity_fact_edge_constructor, *t): t
            for t in tasks
        }

        # ==========================================
        # RESOLUTION: Safe Graph Mutation
        # ==========================================
        for future in as_completed(futures):
            try:
                fact_id, name_to_id_map, edge_info = future.result()

                for relation in edge_info.relations:
                    generated_name = relation.entity_name
                    relation_label = relation.relation_name

                    # Ensure the LLM didn't hallucinate a slightly different name
                    if generated_name in name_to_id_map:
                        mapped_ent_id = name_to_id_map[generated_name]

                        # Safely mutate the graph in the main thread
                        graph.network.add_edge(
                            mapped_ent_id,
                            fact_id,
                            edge_type="entity_fact",
                            label=relation_label
                        )
            except Exception as e:
                # Catching exceptions dynamically per task prevents a single bad prompt from failing the run
                print(f"Error generating entity-fact edges: {e}")

    print("Finished constructing edges between entities and facts.")


def construct_all_edges(graph: Graph, threshold: float = 0.5, max_workers: int = 10):
    if max_workers == 1:
        construct_edges_between_entities_and_facts(graph, entity_batch_size=5)
        construct_edges_between_chunks(graph)
        construct_edges_between_same_chunk_facts(graph)
        construct_edges_between_all_facts(graph, threshold)
    else:
        construct_edges_between_entities_and_facts_parallel(graph, entity_batch_size=5, max_workers=max_workers)
        construct_edges_between_chunks_parallel(graph,max_workers=max_workers)
        construct_edges_between_same_chunk_facts_parallel(graph,max_workers=max_workers)
        construct_edges_between_all_facts_parallel(graph, threshold,max_workers=max_workers)

def construct_edges_during_merge(graph: Graph, threshold: float = 0.5, max_workers: int = 10):
    if max_workers == 1:
        construct_edges_between_chunks(graph)
        construct_edges_between_all_facts(graph, threshold)
    else:
        construct_edges_between_chunks_parallel(graph,max_workers=max_workers)
        construct_edges_between_all_facts_parallel(graph, threshold,max_workers=max_workers)



























