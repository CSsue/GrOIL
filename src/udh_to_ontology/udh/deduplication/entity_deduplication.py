import uuid
import json
import numpy as np
import networkx as nx
import dspy
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from graspologic.partition import hierarchical_leiden

from src.dispergraph_to_ontology.dispergraph.models.Graph import Graph
from src.dispergraph_to_ontology.dispergraph.models.index_objects import EntityFingerprint


# ==========================================
# DSPY SIGNATURES & MODELS
# ==========================================

class MergedEntity(BaseModel):
    duplicate_ids: List[str] = Field(
        description="The exact string IDs of the entities from the input cluster that are duplicates."
    )
    alias_name: str = Field(description="The normalized name for the merged entity.")
    alias_type: str = Field(description="The high-level ontological category (e.g., PERSON, ORGANIZATION).")
    alias_role: str = Field(description="A concise 3-8 word micro-role defining its function.")
    alias_anchors: List[str] = Field(
        description="A combined, deduplicated list of relational anchors from the merged entities.")


class ClusterResolution(dspy.Signature):
    """
    Given a cluster of entity records (JSON), identify subsets that refer to the EXACT same real-world entity.

    For each duplicate group:
    - List their IDs in duplicate_ids.
    - Produce a single normalized alias representing them.

    STRICT RULES:
    - Only merge entities that are unambiguously identical. Different entities that are merely related (e.g., a person and their company) must NOT be merged.
    - If you are unsure whether two entities are the same, do NOT merge them. Err on the side of keeping them separate.
    - Return an empty list immediately if no duplicates exist. Do not explain why.
    - Keep your reasoning to 3 sentences MAXIMUM.
    - Once you have identified all duplicate groups, STOP and return your answer immediately. Do not re-analyze or second-guess.
    - Do not loop. Do not repeat your reasoning. Commit to your answer and output it.
    """
    
    cluster_records: str = dspy.InputField(desc="JSON of entity records with their IDs.")
    merged_entities: List[MergedEntity] = dspy.OutputField(desc="Merged entity groups. Empty list if no duplicates.")


# ==========================================
# DEDUPLICATOR CLASS
# ==========================================

# Default weights for the 3 embedding arrays
SIM_WEIGHTS = {"name": 0.6, "role": 0.3, "type": 0.1}


class GraphDeduplicator:
    def __init__(self, retrieval_model: SentenceTransformer):
        self.retrieval_model = retrieval_model
        self.resolver = dspy.ChainOfThought(ClusterResolution)

    def _build_similarity_graph(self, graph: Graph, top_k: int = 15, threshold: float = 0.75) -> nx.Graph:
        """
        Extracts entities, does 3 separate embedding passes, calculates combined cosine similarity,
        and builds a sparse, UNDIRECTED graph for Leiden clustering.
        """
        print("Building similarity matrix for clustering...")

        entity_ids = list(graph.entities.keys())
        if not entity_ids:
            return nx.Graph()

        names = [graph.entities[eid].name for eid in entity_ids]
        roles = [graph.entities[eid].role for eid in entity_ids]
        types = [graph.entities[eid].type for eid in entity_ids]

        # Three Separate Embedding Passes
        embs_name = self.retrieval_model.encode(names, show_progress_bar=False)
        embs_role = self.retrieval_model.encode(roles, show_progress_bar=False)
        embs_type = self.retrieval_model.encode(types, show_progress_bar=False)

        # Calculate N x N Similarity Matrices
        sim_name = cosine_similarity(embs_name)
        sim_role = cosine_similarity(embs_role)
        sim_type = cosine_similarity(embs_type)

        # Combine based on weights
        sim_total = (
                (SIM_WEIGHTS["name"] * sim_name) +
                (SIM_WEIGHTS["role"] * sim_role) +
                (SIM_WEIGHTS["type"] * sim_type)
        )

        # Build Undirected Graph
        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(entity_ids)

        for i in range(len(entity_ids)):
            # argsort sorts ascending. Take the end of the array, reverse it, skip index 0 (self)
            top_indices = np.argsort(sim_total[i])[-top_k - 1: -1][::-1]

            for j in top_indices:
                score = sim_total[i][j]
                if score >= threshold:
                    sim_graph.add_edge(entity_ids[i], entity_ids[j], weight=float(score))

        return sim_graph

    def _resolve_cluster(self, cluster_ids: List[str], graph: Graph) -> List[Dict[str, Any]]:
        """
        Worker function: Prepares JSON data, calls DSPy, and parses the output
        safely into EntityFingerprint objects.
        """
        # 1. Prepare input payload
        records = []
        for eid in cluster_ids:
            ent = graph.entities[eid]
            records.append({
                "id": eid,
                "name": ent.name,
                "type": ent.type,
                "role": ent.role,
                "anchors": ent.relational_anchors
            })

        payload = json.dumps(records, indent=2)

        # 2. Call LLM
        try:
            result = self.resolver(cluster_records=payload)
            resolved_groups = []

            for merged in result.merged_entities:
                # Security Check: Ensure LLM didn't hallucinate IDs
                valid_dups = [d_id for d_id in merged.duplicate_ids if d_id in cluster_ids]

                # Only return groups that actually merge 2 or more valid items
                if len(valid_dups) > 1:
                    new_fp = EntityFingerprint(
                        name=merged.alias_name,
                        type=merged.alias_type,
                        role=merged.alias_role,
                        relational_anchors=merged.alias_anchors
                    )
                    resolved_groups.append({
                        "alias": new_fp,
                        "duplicates": valid_dups
                    })
            return resolved_groups

        except Exception as e:
            print(f"Error resolving cluster: {e}")
            return []

    def deduplicate(self, graph: Graph, iterations: int = 1, max_workers: int = 15) -> Graph:
        """
        Main entry point. Iteratively clusters, calls LLM workers, and executes
        surgical rewiring on the main thread.
        """
        for iteration in range(iterations):
            print(f"\n--- Deduplication Iteration {iteration + 1} ---")

            # STEP 1: Fast clustering
            sim_graph = self._build_similarity_graph(graph)
            if len(sim_graph.nodes) == 0:
                print("Graph is empty. Skipping deduplication.")
                break

            clusters_raw = hierarchical_leiden(sim_graph, max_cluster_size=64, random_seed=42)

            # Group into list of lists
            clusters = {}
            for node in clusters_raw:
                clusters.setdefault(node.cluster, []).append(node.node)
            cluster_lists = list(clusters.values())

            # Filter out singletons (clusters of size 1 have no duplicates)
            candidate_clusters = [c for c in cluster_lists if len(c) > 1]
            print(f"Found {len(candidate_clusters)} clusters for LLM evaluation.")

            # STEP 2: Parallel LLM Evaluation
            print(f"Executing LLM deduplication across {max_workers} threads...")
            resolution_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._resolve_cluster, cluster, graph): cluster
                    for cluster in candidate_clusters
                }

                for future in as_completed(futures):
                    # Extend the flat list with the groups found by this worker
                    resolution_results.extend(future.result())

            # STEP 3: Surgical Rewiring (Main Thread ONLY)
            print("Surgically rewiring graph edges...")
            deduplicated_count = 0

            for result in resolution_results:
                alias_fp = result["alias"]
                duplicates = result["duplicates"]

                # Double check that we still have nodes to merge
                # (prevents issues if LLM accidentally output the same ID in two different groups)
                duplicates = [d for d in duplicates if graph.network.has_node(d)]
                if len(duplicates) <= 1:
                    continue

                    # A. Create the new node in the graph safely
                # (Generates a new UUID so we don't accidentally overwrite anything)
                alias_id = str(uuid.uuid4())
                graph.entities[alias_id] = alias_fp
                graph.network.add_node(alias_id, type="entity")

                # B. Rewire edges BEFORE deleting old nodes
                for old_node in duplicates:

                    # Rewire incoming edges (Fact -> Entity)
                    for u, _, edge_data in list(graph.network.in_edges(old_node, data=True)):
                        graph.network.add_edge(u, alias_id, **edge_data)

                    # Rewire outgoing edges (Entity -> Fact)
                    for _, v, edge_data in list(graph.network.out_edges(old_node, data=True)):
                        graph.network.add_edge(alias_id, v, **edge_data)

                    # C. Delete the old duplicate
                    graph.network.remove_node(old_node)
                    del graph.entities[old_node]
                    deduplicated_count += 1

            print(f"Removed {deduplicated_count} duplicate nodes in iteration {iteration + 1}.")

        # Optional: Run the label_edges() safety net you just built to ensure edge types are clean
        if hasattr(graph, 'label_edges'):
            graph.label_edges()

        return graph

def main():
    from src.dispergraph_to_ontology.dispergraph.utils.config import configure_dspy
    configure_dspy(max_tokens=35000)

    import warnings
    warnings.filterwarnings("ignore")

    graph = Graph()
    graph.load_graph("./kg_outputs/td.graphml")
    graph.label_edges()
    retreival_model = SentenceTransformer("all-MiniLM-L6-v2")
    deduplicator = GraphDeduplicator(retrieval_model=retreival_model)
    print(f"Before Duplication graph has {len(graph.entities)} entities.")
    graph = deduplicator.deduplicate(graph, max_workers=100)
    print(f"After Deduplication graph has {len(graph.entities)} entities.")
    graph.export_graph("./kg_outputs/td_deduplicated.graphml")


if __name__ == "__main__":
    main()