"""
embeddings.py
-------------
Encode synthetic sentences using a SentenceTransformer model and manage
disk-based caching of the resulting embedding matrices.

The encoder name is baked into the cache filename so that switching models
(e.g., from all-mpnet-base-v2 to all-roberta-large-v1 for robustness checks)
never silently loads stale embeddings.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_encoder(model_name: str = "all-mpnet-base-v2") -> SentenceTransformer:
    """Load a SentenceTransformer encoder by name.

    Parameters
    ----------
    model_name : str
        Any model name accepted by SentenceTransformer, e.g.:
            'all-mpnet-base-v2'       (primary, 768-dim)
            'all-roberta-large-v1'    (robustness check, 1024-dim)
            'all-MiniLM-L12-v2'       (robustness check, 384-dim)

    Returns
    -------
    SentenceTransformer
    """
    return SentenceTransformer(model_name)


def encode_concepts(
    df: pd.DataFrame,
    model: SentenceTransformer,
    cue_col: str = "cue",
    sentence_col: str = "sentence",
    participant_col: str = "participantID",
    cache_path: str = None,
    model_name: str = "model",
    batch_size: int = 512,
) -> dict:
    """Encode synthetic sentences and return a concept-keyed embeddings dict.

    Loads from disk cache if available; encodes and saves otherwise.
    The cache filename encodes the model name so that different encoders
    produce separate cache files.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame with columns: cue_col, sentence_col, participant_col.
    model : SentenceTransformer
        Loaded encoder instance.
    cue_col : str
        Column containing concept labels.
    sentence_col : str
        Column containing synthetic sentences to encode.
    participant_col : str
        Column containing participant identifiers.
    cache_path : str or None
        Directory in which to store/load the cache pickle.
        If None, caching is disabled.
    model_name : str
        Short name used in the cache filename, e.g. 'mpnet', 'roberta'.
    batch_size : int
        Batch size passed to SentenceTransformer.encode().

    Returns
    -------
    dict
        Keys are cue strings. Values are dicts with:
            'matrix'        : np.ndarray of shape (N, D)
            'participantID' : list of participant identifiers (length N)
    """
    cache_file = None
    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        cache_file = os.path.join(cache_path, f"embeddings_{model_name}.pkl")

    if cache_file is not None and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                embeddings = pickle.load(f)
            print(f"Loaded embeddings for {len(embeddings)} concepts from {cache_file}.")
            return embeddings
        except (EOFError, pickle.UnpicklingError):
            print("Cache file corrupt; recomputing embeddings.")
            os.remove(cache_file)

    all_texts = df[sentence_col].tolist()
    all_vecs  = model.encode(all_texts, show_progress_bar=True, batch_size=batch_size)

    embeddings = {}
    for concept in df[cue_col].unique():
        mask = df[cue_col] == concept
        embeddings[concept] = {
            "matrix":        all_vecs[mask.values],
            "participantID": df[mask][participant_col].tolist(),
        }

    if cache_file is not None:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings for {len(embeddings)} concepts to {cache_file}.")

    return embeddings


def subset_embeddings(embeddings: dict, concepts: list) -> dict:
    """Return a sub-dict of embeddings restricted to the given concept list.

    Parameters
    ----------
    embeddings : dict
        Full embeddings dict from encode_concepts().
    concepts : list of str
        Concepts to retain.

    Returns
    -------
    dict
    """
    missing = [c for c in concepts if c not in embeddings]
    if missing:
        print(f"Warning: {len(missing)} concepts not found in embeddings: {missing}")
    return {c: embeddings[c] for c in concepts if c in embeddings}
