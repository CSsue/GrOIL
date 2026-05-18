import numpy as np
from src.dispergraph_to_ontology.dispergraph.utils.math_utils import cosine_similarity_single, max_pooled_list_similarity

# Default Weights
FACT_WEIGHTS = {"sentence": 0.6, "macro": 0.15, "chunk": 0.25}
ENTITY_WEIGHTS = {"name": 0.4, "role": 0.3, "anchors": 0.3}
EDGE_WEIGHTS = {"label": 0.3, "desc": 0.5, "llm_score": 0.2}


def score_fact(
        plan_rewrite_emb: np.ndarray,
        plan_topics_embs: np.ndarray,
        fact_sent_emb: np.ndarray,
        fact_macro_embs: np.ndarray,
        fact_chunk_embs: np.ndarray,
        weights: dict = None
) -> float:
    """
    Scores a single fact against the embedded query plan.
    Safely redistributes weight if macro or chunk topics are missing.
    """
    # Create a local copy of weights so we don't modify the global dictionary
    if weights is None:
        weights = FACT_WEIGHTS.copy()
    else:
        weights = weights.copy()

    # 1. Direct sentence match
    sent_score = cosine_similarity_single(plan_rewrite_emb, fact_sent_emb)

    # 2. Macro Topic Match
    macro_score = 0.0
    if fact_macro_embs is not None and len(fact_macro_embs) > 0:
        macro_score = max_pooled_list_similarity(plan_topics_embs, fact_macro_embs)
    else:
        # If no macro topics, shift the weight to the sentence
        weights["sentence"] += weights["macro"]
        weights["macro"] = 0.0

    # 3. Chunk Topic Match
    chunk_score = 0.0
    if fact_chunk_embs is not None and len(fact_chunk_embs) > 0:
        chunk_score = max_pooled_list_similarity(plan_topics_embs, fact_chunk_embs)
    else:
        # If no chunk topics, shift the weight to the sentence
        weights["sentence"] += weights["chunk"]
        weights["chunk"] = 0.0

    # 4. Weighted Sum
    final_score = (
            (weights["sentence"] * sent_score) +
            (weights["macro"] * macro_score) +
            (weights["chunk"] * chunk_score)
    )
    return final_score


def score_entity(
        plan_target_embs: np.ndarray,
        plan_broad_embs: np.ndarray,
        entity_name_emb: np.ndarray,
        entity_role_emb: np.ndarray,
        entity_anchor_embs: np.ndarray,
        weights: dict = ENTITY_WEIGHTS
) -> float:
    """
    Scores a single entity against the embedded query plan.
    Notice that name and role are passed as 2D arrays of shape (1, D) so they can be treated as lists of length 1.
    """
    name_score = max_pooled_list_similarity(plan_target_embs, entity_name_emb)
    role_score = max_pooled_list_similarity(plan_broad_embs, entity_role_emb)

    # Only score anchors if the entity actually has them
    if entity_anchor_embs is not None and len(entity_anchor_embs) > 0:
        anchor_score = max_pooled_list_similarity(plan_target_embs, entity_anchor_embs)
    else:
        # If no anchors exist, we redistribute the anchor weight to the name and role
        # to prevent penalizing entities that inherently lack anchors.
        anchor_score = 0.0
        redistribute = weights["anchors"] / 2
        weights = {
            "name": weights["name"] + redistribute,
            "role": weights["role"] + redistribute,
            "anchors": 0.0
        }

    final_score = (
            (weights["name"] * name_score) +
            (weights["role"] * role_score) +
            (weights["anchors"] * anchor_score)
    )
    return final_score

# Notice how the weights for each dictionary sum perfectly to 1.0
FF_EDGE_WEIGHTS = {"label": 0.3, "desc": 0.5, "llm_score": 0.2}
EF_EDGE_WEIGHTS = {"broad_anchors": 0.5, "semantics": 0.5}
FE_EDGE_WEIGHTS = {"broad_anchors": 1.0}

def score_fact_fact_edge(
        plan_labels_embs: np.ndarray,
        plan_semantics_embs: np.ndarray,
        edge_label_emb: np.ndarray,
        edge_desc_emb: np.ndarray,
        edge_llm_relevance: float,
        weights: dict = FF_EDGE_WEIGHTS
) -> float:
    """Scores Fact -> Fact narrative edges."""
    label_score = max_pooled_list_similarity(plan_labels_embs, edge_label_emb)
    semantic_score = max_pooled_list_similarity(plan_semantics_embs, edge_desc_emb)

    return (
            (weights["label"] * label_score) +
            (weights["desc"] * semantic_score) +
            (weights["llm_score"] * edge_llm_relevance)
    )


def score_entity_fact_edge(
        plan_broad_embs: np.ndarray,
        plan_semantics_embs: np.ndarray,
        edge_label_emb: np.ndarray,
        weights: dict = EF_EDGE_WEIGHTS
) -> float:
    """
    Scores Entity -> Fact functional edges (e.g., 'STARS_IN_MOVIE').
    Scores the single edge label against both the expected roles and semantics.
    """
    role_score = max_pooled_list_similarity(plan_broad_embs, edge_label_emb)
    semantic_score = max_pooled_list_similarity(plan_semantics_embs, edge_label_emb)

    return (
            (weights["broad_anchors"] * role_score) +
            (weights["semantics"] * semantic_score)
    )


def score_fact_entity_edge(
        plan_broad_embs: np.ndarray,
        edge_label_emb: np.ndarray,
        weights: dict = FE_EDGE_WEIGHTS
) -> float:
    """
    Scores Fact -> Entity ontological edges (e.g., 'PERSON', 'ORGANIZATION').
    Checks if the edge points to the type of entity the user is looking for.
    """
    type_score = max_pooled_list_similarity(plan_broad_embs, edge_label_emb)

    return weights["broad_anchors"] * type_score