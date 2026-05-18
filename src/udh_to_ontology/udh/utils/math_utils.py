import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_single(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two single embedding vectors.
    Expects 1D or 2D arrays of a single vector.
    """
    # Reshape to 2D for sklearn (1, D)
    e1 = np.atleast_2d(emb1)
    e2 = np.atleast_2d(emb2)
    return float(cosine_similarity(e1, e2)[0][0])


def max_pooled_list_similarity(query_embs: np.ndarray, target_embs: np.ndarray) -> float:
    """
    Computes the S_list similarity between a list of query embeddings and target embeddings.
    """
    # ADD THIS SAFETY CHECK:
    if query_embs is None or target_embs is None:
        return 0.0

    if len(query_embs) == 0 or len(target_embs) == 0:
        return 0.0

    sim_matrix = cosine_similarity(query_embs, target_embs)
    max_sims = np.max(sim_matrix, axis=1)
    return float(np.mean(max_sims))


def compute_mmr(
        candidate_scores: list[float],
        candidate_embeddings: np.ndarray,
        top_k: int,
        lambda_param: float = 0.5
) -> list[int]:
    """
    Selects top_k items using Maximal Marginal Relevance (MMR).
    Balances relevance (candidate_scores) with diversity (candidate_embeddings).

    lambda_param: 1.0 means pure relevance (standard top-k). 0.0 means pure diversity.
    Returns: A list of indices corresponding to the selected candidates.
    """
    if len(candidate_scores) == 0:
        return []

    # Ensure inputs are numpy arrays for fast math
    scores = np.array(candidate_scores)
    embs = np.atleast_2d(candidate_embeddings)

    # Handle cases where we ask for more K than we have candidates
    top_k = min(top_k, len(scores))

    selected_indices = []
    unselected_indices = list(range(len(scores)))

    # Pre-compute the NxN similarity matrix between ALL candidates
    # This prevents recalculating cosine similarity in the loop
    sim_matrix = cosine_similarity(embs)

    for _ in range(top_k):
        if not selected_indices:
            # First iteration: just pick the one with the highest raw relevance score
            best_idx = unselected_indices[np.argmax(scores[unselected_indices])]
        else:
            # Calculate MMR score for all unselected items
            best_mmr_score = -np.inf
            best_idx = -1

            for idx in unselected_indices:
                # 1. Relevance: The raw score passed into the function
                relevance = scores[idx]

                # 2. Penalty: Max similarity to items already selected
                # (How redundant is this item compared to what we already chose?)
                redundancy_penalty = np.max(sim_matrix[idx, selected_indices])

                # 3. MMR Equation
                mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * redundancy_penalty)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

    return selected_indices