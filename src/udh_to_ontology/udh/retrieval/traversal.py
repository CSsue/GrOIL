import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

from src.dispergraph_to_ontology.dispergraph.models.Graph import Graph
from src.dispergraph_to_ontology.dispergraph.utils.math_utils import compute_mmr
from src.dispergraph_to_ontology.dispergraph.retrieval.scoring import score_fact_fact_edge, score_entity_fact_edge, score_fact_entity_edge

class Path:
    """Tracks the state of a traversal branch to prevent loops and monitor depth."""

    def __init__(self, start_node_id: str, is_fact: bool):
        self.current_node = start_node_id
        self.visited_nodes = {start_node_id}
        self.fact_depth = 1 if is_fact else 0

        # History stores the path taken.
        # Format: list of tuples (source_id, edge_type, edge_data, target_id)
        self.history = []

    def branch(self, target_node_id: str, is_fact: bool, edge_type: str, edge_data: dict) -> 'Path':
        """Creates a new Path object representing a step forward."""
        new_path = deepcopy(self)
        new_path.visited_nodes.add(target_node_id)
        new_path.current_node = target_node_id
        new_path.history.append((self.current_node, edge_type, edge_data, target_node_id))

        if is_fact:
            new_path.fact_depth += 1

        return new_path


# ==========================================
# STRING ASSEMBLY HELPERS
# ==========================================

def _get_node_representation(graph: Graph, node_id: str) -> str:
    """Returns the formatted string representation for a Fact or Entity."""
    if node_id in graph.facts:
        return f'"{graph.facts[node_id].sentence}"'
    elif node_id in graph.entities:
        ent = graph.entities[node_id]
        return f"[{ent.name} ({ent.role})]"
    return "[Unknown Node]"


def assemble_path_string(graph: Graph, path: Path) -> str:
    """
    Translates a Path object's history into a highly readable narrative string
    for the Cross-Encoder to evaluate.
    """
    if not path.history:
        return _get_node_representation(graph, path.current_node)

    # Start with the representation of the very first node
    first_node_id = path.history[0][0]
    assembled_str = _get_node_representation(graph, first_node_id)

    for _, edge_type, edge_data, target_id in path.history:
        target_rep = _get_node_representation(graph, target_id)

        if edge_type == "fact_fact":
            desc = edge_data.get("description", "is related to")
            assembled_str += f" -> {desc} -> {target_rep}"

        elif edge_type == "fact_entity":
            label = edge_data.get("label", "has property")
            assembled_str += f" -> (has {label}) -> {target_rep}"

        elif edge_type == "entity_fact":
            label = edge_data.get("label", "participates in")
            assembled_str += f" -> (participates in event: {label}) -> {target_rep}"

    return assembled_str


# ==========================================
# TRAVERSAL & EXPANSION
# ==========================================

def expand_paths(
        engine,  # The Retriever engine (holds graph, embeddings, and planner outputs)
        seeds: List[str],
        plan: object,
        max_depth: int = 3,
        beam_width: int = 3
) -> List[Path]:
    """
    Executes a Bounded Beam Search with Local MMR to explore the graph.
    """
    graph = engine.graph
    active_paths = []
    completed_paths = []

    # Initialize Active Paths
    for seed in seeds:
        is_fact = seed in graph.facts
        active_paths.append(Path(start_node_id=seed, is_fact=is_fact))

    # Pre-fetch the query plan embeddings for scoring edges
    plan_labels_embs = engine.retrieval_model.encode(plan.target_edge_labels, show_progress_bar=False) if plan.target_edge_labels else np.array(
        [])
    plan_semantics_embs = engine.retrieval_model.encode(
        plan.target_edge_semantics,show_progress_bar=False) if plan.target_edge_semantics else np.array([])
    plan_broad_embs = engine.retrieval_model.encode(plan.broad_anchors,show_progress_bar=False) if plan.broad_anchors else np.array([])

    while active_paths:
        next_active = []

        for path in active_paths:
            current_id = path.current_node

            # Look up outgoing edges
            outgoing_edges = list(graph.network.out_edges(current_id, data=True))

            valid_candidates = []
            candidate_scores = []
            candidate_diversity_embs = []  # Used for Local MMR

            for u, v, edge_data in outgoing_edges:
                edge_type = edge_data.get("edge_type")

                # Filter 1: Loop prevention
                if v in path.visited_nodes:
                    continue

                # Filter 2: Ignore irrelevant edge types (like chunk_fact)
                if edge_type not in ["fact_fact", "entity_fact", "fact_entity"]:
                    continue

                edge_tuple = (u, v)
                score = 0.0
                diversity_emb = None

                # Score based on edge type using the engine's pre-computed embeddings
                if edge_type == "fact_fact":
                    score = score_fact_fact_edge(
                        plan_labels_embs, plan_semantics_embs,
                        engine.ff_label_embs[edge_tuple],
                        engine.ff_desc_embs[edge_tuple],
                        edge_data.get("score", 0.0)
                    )
                    diversity_emb = engine.ff_desc_embs[edge_tuple][0]  # Use desc for diversity

                elif edge_type == "entity_fact":
                    score = score_entity_fact_edge(
                        plan_broad_embs, plan_semantics_embs,
                        engine.ef_label_embs[edge_tuple]
                    )
                    diversity_emb = engine.ef_label_embs[edge_tuple][0]

                elif edge_type == "fact_entity":
                    score = score_fact_entity_edge(
                        plan_broad_embs,
                        engine.fe_label_embs[edge_tuple]
                    )
                    diversity_emb = engine.fe_label_embs[edge_tuple][0]

                valid_candidates.append((v, edge_type, edge_data))
                candidate_scores.append(score)
                candidate_diversity_embs.append(diversity_emb)

            # Handle Dead Ends
            if not valid_candidates:
                completed_paths.append(path)
                continue

            # Local MMR Execution (Select top 'beam_width' edges)
            selected_edge_indices = compute_mmr(
                candidate_scores=candidate_scores,
                candidate_embeddings=np.array(candidate_diversity_embs),
                top_k=beam_width,
                lambda_param=0.5  # Balance local relevance and local diversity
            )

            # Branch the path for the selected edges
            for idx in selected_edge_indices:
                target_node, e_type, e_data = valid_candidates[idx]
                is_fact = target_node in graph.facts

                new_path = path.branch(target_node, is_fact, e_type, e_data)

                # Check Depth
                if new_path.fact_depth >= max_depth:
                    completed_paths.append(new_path)
                else:
                    next_active.append(new_path)

        active_paths = next_active

    return completed_paths


# ==========================================
# FINAL GLOBAL RANKING
# ==========================================

def rank_paths_global(
        engine,
        query: str,
        completed_paths: List[Path],
        cross_encoder: CrossEncoder,
        final_top_k: int = 5,
        verbose=False
) -> Tuple[List[str], List[Path]]:
    """
    Assembles the paths into text, scores them against the query using a Cross-Encoder,
    and returns the final diverse set of logical answers using Global MMR.
    """
    if not completed_paths:
        return [], []

    # 1. Assemble Paths
    assembled_strings = [assemble_path_string(engine.graph, p) for p in completed_paths]

    # 2. Score with Cross Encoder (Relevance)
    ce_inputs = [[query, path_str] for path_str in assembled_strings]
    if verbose:
        print(f"Running Cross-Encoder on {len(ce_inputs)} assembled paths...")
    ce_scores = cross_encoder.predict(ce_inputs)

    # 3. Embed the assembled strings (Diversity)
    # This ensures we don't return 5 paths that are essentially identical variations of each other.
    path_embeddings = engine.retrieval_model.encode(assembled_strings, show_progress_bar=False)

    # 4. Global MMR Execution
    selected_indices = compute_mmr(
        candidate_scores=list(ce_scores),
        candidate_embeddings=path_embeddings,
        top_k=final_top_k,
        lambda_param=0.6  # Lean slightly towards relevance for the final output
    )

    final_strings = [assembled_strings[i] for i in selected_indices]
    final_paths = [completed_paths[i] for i in selected_indices]

    return final_strings, final_paths