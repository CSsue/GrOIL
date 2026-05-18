import numpy as np
import networkx as nx
import dspy
from sentence_transformers import SentenceTransformer

from src.dispergraph_to_ontology.dispergraph.models.Graph import Graph
from src.dispergraph_to_ontology.dispergraph.retrieval.planner import QueryPlan  # Assuming you wrapped the signature in a module here
from src.dispergraph_to_ontology.dispergraph.retrieval.scoring import score_fact, score_entity
from src.dispergraph_to_ontology.dispergraph.utils.math_utils import compute_mmr
from .traversal import *


class Retriever:
    def __init__(
            self,
            retrieval_model: SentenceTransformer,
            cross_encoder: CrossEncoder,  # Added Cross-Encoder
            graph: Graph = None,
            graph_path: str = None
    ):
        """
        Initializes the retriever with embedding models and the knowledge graph.
        """
        self.retrieval_model = retrieval_model
        self.cross_encoder = cross_encoder

        if graph:
            self.graph = graph
        elif graph_path:
            self.graph = Graph()
            self.graph.load_graph(graph_path)
            self.graph.label_edges()
        else:
            raise ValueError("Must provide either a graph object or a graph_path.")

        # Initialize DSPy Query Planner
        self.planner = dspy.ChainOfThought(QueryPlan)

        # Node embeddings
        self.entity_embs = {}
        self.fact_embs = {}

        # Edge embeddings mapped by (source_id, target_id)
        self.ff_desc_embs = {}
        self.ff_label_embs = {}
        self.ef_label_embs = {}
        self.fe_label_embs = {}

        self._initialize_embeddings()
        self._initialize_edge_embeddings()

    def _initialize_embeddings(self, verbose=False):
        """
        Solves the Flatten-Unflatten problem. Gathers every string from every object,
        encodes them in one massive batch, and maps the resulting vectors back to the Node IDs.
        """
        if verbose:
            print("Flattening graph data for embedding...")

        flat_strings = []
        tracking = {}  # Maps (node_id, field) -> (start_idx, end_idx)
        current_idx = 0

        # --- Flatten Entities ---
        for ent_id, ent in self.graph.entities.items():
            # Name
            flat_strings.append(ent.name)
            tracking[(ent_id, "name")] = (current_idx, current_idx + 1)
            current_idx += 1

            # Role
            flat_strings.append(ent.role)
            tracking[(ent_id, "role")] = (current_idx, current_idx + 1)
            current_idx += 1

            # Relational Anchors (List)
            if ent.relational_anchors:
                num_anchors = len(ent.relational_anchors)
                flat_strings.extend(ent.relational_anchors)
                tracking[(ent_id, "anchors")] = (current_idx, current_idx + num_anchors)
                current_idx += num_anchors
            else:
                tracking[(ent_id, "anchors")] = None

        # --- Flatten Facts ---
        for fact_id, fact in self.graph.facts.items():
            # Sentence
            flat_strings.append(fact.sentence)
            tracking[(fact_id, "sentence")] = (current_idx, current_idx + 1)
            current_idx += 1

            # Macro Topics (List)
            if fact.macro_topics:
                num_mac = len(fact.macro_topics)
                flat_strings.extend(fact.macro_topics)
                tracking[(fact_id, "macro")] = (current_idx, current_idx + num_mac)
                current_idx += num_mac
            else:
                tracking[(fact_id, "macro")] = None

            # Chunk Topics (List)
            if fact.chunk_topics:
                num_chk = len(fact.chunk_topics)
                flat_strings.extend(fact.chunk_topics)
                tracking[(fact_id, "chunk")] = (current_idx, current_idx + num_chk)
                current_idx += num_chk
            else:
                tracking[(fact_id, "chunk")] = None

        # --- Massive Batch Encoding ---
        if verbose:
            print(f"Encoding {len(flat_strings)} total text segments...")
        # model.encode returns an (N, D) numpy array
        all_embs = self.retrieval_model.encode(flat_strings, show_progress_bar=True)

        # --- Unflatten into Dictionaries ---
        if verbose:
            print("Re-mapping embeddings to graph nodes...")
        for ent_id in self.graph.entities.keys():
            n_slice = tracking[(ent_id, "name")]
            r_slice = tracking[(ent_id, "role")]
            a_slice = tracking[(ent_id, "anchors")]

            self.entity_embs[ent_id] = {
                "name": all_embs[n_slice[0]:n_slice[1]],
                "role": all_embs[r_slice[0]:r_slice[1]],
                "anchors": all_embs[a_slice[0]:a_slice[1]] if a_slice else None
            }

        for fact_id in self.graph.facts.keys():
            s_slice = tracking[(fact_id, "sentence")]
            m_slice = tracking[(fact_id, "macro")]
            c_slice = tracking[(fact_id, "chunk")]

            self.fact_embs[fact_id] = {
                "sentence": all_embs[s_slice[0]:s_slice[1]],
                "macro": all_embs[m_slice[0]:m_slice[1]] if m_slice else None,
                "chunk": all_embs[c_slice[0]:c_slice[1]] if c_slice else None
            }

    def _initialize_edge_embeddings(self, verbose=False):
        """
        Gathers and embeds edge descriptions and labels.
        Uses separate passes for cleanly mapping back to (source, target) tuples.
        """
        if verbose:
            print("Gathering edges for embedding...")

        # Temporary lists to hold the (u, v) tuples and the strings to embed
        ff_edges, ff_descs, ff_labels = [], [], []
        ef_edges, ef_labels = [], []
        fe_edges, fe_labels = [], []

        # 1. Iterate through all edges in the NetworkX graph
        for u, v, data in self.graph.network.edges(data=True):

            # Fact -> Fact Edges
            if u in self.graph.facts and v in self.graph.facts:
                ff_edges.append((u, v))
                ff_descs.append(data.get("description", ""))
                ff_labels.append(data.get("label", ""))

            # Entity -> Fact Edges
            elif data.get("edge_type") == "entity_fact":
                ef_edges.append((u, v))
                ef_labels.append(data.get("label", ""))

            # Fact -> Entity Edges
            elif data.get("edge_type") == "fact_entity":
                fe_edges.append((u, v))
                fe_labels.append(data.get("label", ""))

        # 2. Separate Encoding Passes
        # Using separate passes is perfectly fine here and keeps the mapping logic incredibly simple.
        if verbose:
            print(f"Encoding {len(ff_edges)} Fact-Fact edges (Descriptions & Labels)...")
        if ff_edges:
            ff_desc_matrix = self.retrieval_model.encode(ff_descs, show_progress_bar=True)
            ff_label_matrix = self.retrieval_model.encode(ff_labels, show_progress_bar=True)

        if verbose:
            print(f"Encoding {len(ef_edges)} Entity-Fact edge labels...")
        if ef_edges:
            ef_label_matrix = self.retrieval_model.encode(ef_labels, show_progress_bar=True)

        if verbose:
            print(f"Encoding {len(fe_edges)} Fact-Entity edge labels...")
        if fe_edges:
            fe_label_matrix = self.retrieval_model.encode(fe_labels, show_progress_bar=True)

        # 3. Map back to Dictionaries
        if verbose:
            print("Mapping edge embeddings back to (source, target) tuples...")

        # CRITICAL NumPy TRICK: We slice using [i:i+1] instead of [i].
        # If we use [i], numpy returns a 1D array of shape (D,).
        # If we use [i:i+1], numpy returns a 2D array of shape (1, D).
        # Our `max_pooled_list_similarity` function strictly requires 2D arrays!

        for i, edge in enumerate(ff_edges):
            self.ff_desc_embs[edge] = ff_desc_matrix[i:i + 1]
            self.ff_label_embs[edge] = ff_label_matrix[i:i + 1]

        for i, edge in enumerate(ef_edges):
            self.ef_label_embs[edge] = ef_label_matrix[i:i + 1]

        for i, edge in enumerate(fe_edges):
            self.fe_label_embs[edge] = fe_label_matrix[i:i + 1]

    def get_seeds(self, query: str, top_k: int = 10, lambda_mmr: float = 0.6, verbose=False) -> tuple[list[str], object]:
        """
        Executes the query plan, scores all facts and entities independently,
        and returns the top-k diverse seeds using MMR.
        """
        # 1. Run Query Planner
        if verbose:
            print(f"Planning query strategy for: '{query}'")
        plan = self.planner(user_query=query)

        # 2. Embed the Query Plan components
        # Note: If the LLM returns empty lists, encode returns an empty array,
        # which our max_pooled_list_similarity handles gracefully (returns 0.0)
        plan_rewrite_emb = self.retrieval_model.encode([plan.rewritten_query], show_progress_bar=False)
        plan_targets_embs = self.retrieval_model.encode(plan.target_entities,show_progress_bar=False) if plan.target_entities else np.array([])
        plan_broad_embs = self.retrieval_model.encode(plan.broad_anchors,show_progress_bar=False) if plan.broad_anchors else np.array([])
        plan_topics_embs = self.retrieval_model.encode(plan.target_topics,show_progress_bar=False) if plan.target_topics else np.array([])

        candidate_ids = []
        candidate_scores = []
        candidate_diversity_embs = []  # Used exclusively for MMR redundancy checks

        # 3. Score all Entities
        for ent_id, embs in self.entity_embs.items():
            score = score_entity(
                plan_target_embs=plan_targets_embs,
                plan_broad_embs=plan_broad_embs,
                entity_name_emb=embs["name"],
                entity_role_emb=embs["role"],
                entity_anchor_embs=embs["anchors"]
            )
            candidate_ids.append(ent_id)
            candidate_scores.append(score)
            # Use the entity's name embedding to represent it during the MMR diversity check
            candidate_diversity_embs.append(embs["name"][0])

        # 4. Score all Facts
        for fact_id, embs in self.fact_embs.items():
            score = score_fact(
                plan_rewrite_emb=plan_rewrite_emb,
                plan_topics_embs=plan_topics_embs,
                fact_sent_emb=embs["sentence"],
                fact_macro_embs=embs["macro"],
                fact_chunk_embs=embs["chunk"]
            )
            candidate_ids.append(fact_id)
            candidate_scores.append(score)
            # Use the fact's sentence embedding to represent it during the MMR diversity check
            candidate_diversity_embs.append(embs["sentence"][0])

        # 5. Apply MMR (Maximal Marginal Relevance)
        selected_indices = compute_mmr(
            candidate_scores=candidate_scores,
            candidate_embeddings=np.array(candidate_diversity_embs),
            top_k=top_k,
            lambda_param=lambda_mmr
        )

        # Map indices back to the actual Node IDs
        seeds = [candidate_ids[i] for i in selected_indices]

        if verbose:
            print(f"Selected {len(seeds)} diverse seeds.")
        return seeds, plan

    def retrieve(
            self,
            query: str,
            top_k_seeds: int = 8,
            max_depth: int = 3,
            beam_width: int = 3,
            final_top_k: int = 5,
            return_raw_paths: bool = False,
            verbose=False
    ) -> str:
        """
        The main public API for the Retriever.
        Takes a natural language query and returns a single formatted context string
        containing the most logically sound traversal paths.
        """
        if verbose:
            print(f"\n--- Starting Retrieval Pipeline for: '{query}' ---")

        # Step 1: Query Planning & Seed Selection
        seeds, plan = self.get_seeds(query, top_k=top_k_seeds)

        if not seeds:
            print("No relevant seeds found.")
            return ""

        # Step 2: Bounded Beam Search (Local MMR Edge Expansion)
        if verbose:
            print(f"Expanding {len(seeds)} seeds to a max fact depth of {max_depth}...")
        completed_paths = expand_paths(
            engine=self,
            seeds=seeds,
            plan=plan,
            max_depth=max_depth,
            beam_width=beam_width
        )

        if verbose:
            print(f"Graph traversal generated {len(completed_paths)} candidate paths.")

        # Step 3: Global Ranking (Cross-Encoder + Diversity MMR)
        final_strings, final_paths = rank_paths_global(
            engine=self,
            query=query,
            completed_paths=completed_paths,
            cross_encoder=self.cross_encoder,
            final_top_k=final_top_k
        )

        # Step 4: Final Context Formatting
        # Join the top-K narrative strings with double newlines
        context_string = "\n\n".join(final_strings)

        if verbose:
            print("--- Retrieval Complete ---")

        if return_raw_paths:
            # Useful if you need the actual Node IDs downstream
            return context_string, final_paths

        return context_string

def main():
    from src.dispergraph_to_ontology.dispergraph.utils.config import configure_dspy
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from src.dispergraph_to_ontology.dispergraph.retrieval.engine import Retriever

    configure_dspy(max_tokens=30000)

    # 1. Load your models
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 2. Initialize the Retriever
    # (This does the heavy lifting of flattening and embedding the graph)
    retriever = Retriever(
        retrieval_model=bi_encoder,
        cross_encoder=cross_encoder,
        graph_path="./my_knowledge_graph.graphml"
    )

    # 3. Retrieve Context
    user_question = "What are the liability limits for the main policyholder's vehicle?"
    context = retriever.retrieve(
        query=user_question,
        top_k_seeds=10,
        max_depth=2,  # How many 'Facts' deep a path should go
        beam_width=3,  # How many branches to explore per node
        final_top_k=4  # How many final paragraphs to return
    )

    # 4. Feed `context` to your Generation LLM!
    print("Context to inject into generation prompt:\n")
    print(context)

if __name__ == "__main__":
    main()
