"""
sharedness.py
-------------
Compute the sharedness estimand S_b for each concept in an embeddings dict.

S_b is defined as 1 - delta_bar_b, where delta_bar_b is the mean pairwise
cosine dissimilarity across all unique participant dyads (upper triangle, k=1).
Dissimilarity uses the angular normalization delta = (1 - cos) / 2, which
bounds the metric to [0, 1].

High S_b indicates that participants' associations converge in embedding space;
low S_b indicates representational divergence across the population.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_sharedness(embeddings: dict, concepts: list = None) -> pd.DataFrame:
    """Compute S_b for each concept via mean pairwise cosine dissimilarity.

    Parameters
    ----------
    embeddings : dict
        Concept-keyed dict from encode_concepts(). Each value must contain
        a 'matrix' key with an (N, D) float array.
    concepts : list of str or None
        Subset of concepts to process. If None, all keys in embeddings
        are processed.

    Returns
    -------
    pd.DataFrame
        Columns: concept, N, delta_bar_b, S_b.
        Sorted by S_b descending.
    """
    if concepts is None:
        concepts = list(embeddings.keys())

    rows = []

    for concept in tqdm(concepts, desc="Computing sharedness"):
        if concept not in embeddings:
            continue

        emb       = embeddings[concept]["matrix"]
        valid_idx = np.where(np.linalg.norm(emb, axis=1) > 0)[0]
        emb_v     = emb[valid_idx]
        N         = len(valid_idx)

        if N < 2:
            continue

        # Normalize rows to unit length, then compute full cosine similarity matrix
        norms = np.linalg.norm(emb_v, axis=1, keepdims=True)
        emb_n = emb_v / norms
        cos   = emb_n @ emb_n.T

        # Angular dissimilarity: delta = (1 - cos) / 2
        delta = (1 - cos) / 2

        i_idx, j_idx = np.triu_indices(N, k=1)
        d_vec        = delta[i_idx, j_idx]

        delta_bar = float(np.mean(d_vec))
        S_b       = 1.0 - delta_bar

        rows.append({"concept": concept, "N": N, "delta_bar_b": delta_bar, "S_b": S_b})

    return (
        pd.DataFrame(rows)
        .sort_values("S_b", ascending=False)
        .reset_index(drop=True)
    )


def compute_pairwise_distances(embeddings: dict, concept: str) -> np.ndarray:
    """Return the upper-triangle pairwise dissimilarity vector for one concept.

    Useful for downstream permutation tests or distribution comparisons.

    Parameters
    ----------
    embeddings : dict
    concept : str

    Returns
    -------
    np.ndarray of shape (N*(N-1)/2,)
    """
    emb       = embeddings[concept]["matrix"]
    valid_idx = np.where(np.linalg.norm(emb, axis=1) > 0)[0]
    emb_v     = emb[valid_idx]
    norms     = np.linalg.norm(emb_v, axis=1, keepdims=True)
    emb_n     = emb_v / norms
    cos       = emb_n @ emb_n.T
    delta     = (1 - cos) / 2
    i_idx, j_idx = np.triu_indices(len(emb_v), k=1)
    return delta[i_idx, j_idx]
