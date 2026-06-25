"""
llm.py
------
Elicit word associations from a large language model and compare the resulting
embedding distributions to human participant distributions.

The elicitation pipeline mirrors the SWOW sentence construction exactly:
the LLM is prompted for three associates per concept, which are assembled into
the same synthetic sentence format used for human participants. This ensures
that LLM and human embeddings are directly comparable in the same space.

The machine behavior demonstration tests for LLM hypercentrality: the hypothesis
that LLM associations cluster closer to the human population mean than humans do
to each other, reflecting a collapse of representational variance under RLHF.
"""

import re
import time
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _make_provider_client(provider: str, api_key: str, model_name: str,
                          system_prompt: str, temperature: float):
    """Build a (generate_fn, blocked_check_fn) pair for the given provider.

    generate_fn(user_message) -> raw text response, or raises on API error.
    blocked_check_fn(response) -> True if the response was blocked by a
    safety filter (only meaningful for Gemini; always False otherwise).

    Each branch imports its SDK lazily so only the provider actually used
    needs to be installed.
    """
    provider = provider.lower()

    if provider == "gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for provider='gemini'. "
                "Install with: pip install google-generativeai"
            )
        genai.configure(api_key=api_key)
        llm_model = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature, max_output_tokens=300
            ),
        )

        def generate_fn(user_message):
            response = llm_model.generate_content(user_message)
            if not response.candidates or response.candidates[0].finish_reason == 3:
                return None  # blocked
            return response.text

        return generate_fn

    elif provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for provider='openai'. "
                "Install with: pip install openai"
            )
        client = OpenAI(api_key=api_key)

        def generate_fn(user_message):
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            choice = response.choices[0]
            if choice.finish_reason == "content_filter":
                return None  # blocked
            return choice.message.content

        return generate_fn

    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required for provider='anthropic'. "
                "Install with: pip install anthropic"
            )
        client = anthropic.Anthropic(api_key=api_key)

        def generate_fn(user_message):
            response = client.messages.create(
                model=model_name,
                max_tokens=300,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            if response.stop_reason == "refusal":
                return None  # blocked
            return response.content[0].text

        return generate_fn

    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Choose 'gemini', 'openai', or 'anthropic'."
        )


def collect_llm_associations(
    concepts: list,
    api_key: str,
    provider: str = "gemini",
    k: int = 20,
    temperature: float = 0.7,
    model_name: str = "gemini-2.5-flash",
    sleep_sec: float = 0.3,
    output_dir: str = "llm_associations",
    system_prompt: str = None,
    user_prompt: str = None,
) -> pd.DataFrame:
    """Elicit K association triples per concept from an LLM.

    Each call returns three associates (R1, R2, R3) which are assembled into
    a synthetic sentence using the same format as the SWOW pipeline. Responses
    blocked by the model's safety filters are flagged and excluded downstream.

    Supports three providers, switched via the `provider` argument. Each
    provider's SDK is imported lazily, so only the package for the provider
    actually used needs to be installed:
      'gemini'    -> google-generativeai   (pip install google-generativeai)
      'openai'    -> openai                (pip install openai)
      'anthropic' -> anthropic             (pip install anthropic)

    Parameters
    ----------
    concepts : list of str
        Concept words to elicit associations for.
    api_key : str
        API key for the chosen provider.
    provider : str
        'gemini', 'openai', or 'anthropic'.
    k : int
        Number of independent association triples to elicit per concept.
    temperature : float
        Sampling temperature. 0.7 is recommended to introduce variability
        while keeping responses coherent.
    model_name : str
        Model identifier for the chosen provider, e.g. 'gemini-2.5-flash',
        'gpt-4o-mini', 'claude-haiku-4-5-20251001'.
    sleep_sec : float
        Pause between API calls to avoid rate limiting.
    output_dir : str
        Directory in which to save the raw CSV output.
    system_prompt : str or None
        Override the default system prompt.
    user_prompt : str or None
        Override the default user prompt. Must contain {concept} placeholder.

    Returns
    -------
    pd.DataFrame
        Columns: cue, run_id, R1, R2, R3, sentence, valid, blocked.
    """
    _system = system_prompt or (
        "You are participating in a word association study. "
        "Respond only with the requested words, nothing else."
    )
    _user = user_prompt or (
        "Respond with the first three words that come to mind when you see "
        "the word {concept}. Output format: word1, word2, word3. Nothing else."
    )

    generate_fn = _make_provider_client(provider, api_key, model_name, _system, temperature)

    rows = []

    for concept in concepts:
        print(f"\n{concept}")
        n_valid, n_blocked = 0, 0

        for run_id in range(k):
            try:
                raw = generate_fn(_user.format(concept=concept))

                if raw is None:
                    rows.append({
                        "cue": concept, "run_id": run_id,
                        "R1": None, "R2": None, "R3": None,
                        "sentence": None, "valid": False, "blocked": True,
                    })
                    print(f"  {run_id:02d}: BLOCKED")
                    n_blocked += 1
                    time.sleep(sleep_sec)
                    continue

                associates = _parse_associates(raw)
                valid      = associates is not None
                rows.append({
                    "cue":      concept,
                    "run_id":   run_id,
                    "R1":       associates[0] if valid else None,
                    "R2":       associates[1] if valid else None,
                    "R3":       associates[2] if valid else None,
                    "sentence": _build_sentence(concept, associates) if valid else None,
                    "valid":    valid,
                    "blocked":  False,
                })
                if valid:
                    n_valid += 1
                print(f"  {run_id:02d}: {associates or 'PARSE ERROR: ' + repr(raw)}")

            except Exception as e:
                rows.append({
                    "cue": concept, "run_id": run_id,
                    "R1": None, "R2": None, "R3": None,
                    "sentence": None, "valid": False, "blocked": False, "error": str(e),
                })
                print(f"  {run_id:02d}: API ERROR: {e}")

            time.sleep(sleep_sec)

        print(f"  {n_valid}/{k} valid, {n_blocked} blocked")

    df = pd.DataFrame(rows)

    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(Path(output_dir) / f"associations_{timestamp}.csv", index=False)
    print(f"\nSaved raw associations to {output_dir}/associations_{timestamp}.csv")

    return df


def encode_llm_associations(
    df_llm: pd.DataFrame,
    model,
    batch_size: int = 512,
    cache_path: str = None,
) -> dict:
    """Encode valid LLM association sentences and return an embeddings dict.

    The returned dict has the same structure as the human embeddings dict
    from encode_concepts(), with 'run_id' in place of 'participantID'.

    Parameters
    ----------
    df_llm : pd.DataFrame
        Output of collect_llm_associations(). Only rows with valid=True
        are encoded.
    model : SentenceTransformer
        Must be the same encoder used for human embeddings.
    batch_size : int
    cache_path : str or None
        If provided, saves/loads the dict as llm_embeddings.pkl in that directory.

    Returns
    -------
    dict
        Keys are concept strings. Values contain 'matrix' and 'run_id'.
    """
    cache_file = None
    if cache_path is not None:
        import os
        os.makedirs(cache_path, exist_ok=True)
        cache_file = os.path.join(cache_path, "llm_embeddings.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                embeddings_llm = pickle.load(f)
            print(f"Loaded LLM embeddings for {len(embeddings_llm)} concepts from {cache_file}.")
            return embeddings_llm

    df_valid      = df_llm[df_llm["valid"]].copy().reset_index(drop=True)
    all_vecs      = model.encode(df_valid["sentence"].tolist(),
                                 show_progress_bar=True, batch_size=batch_size)

    embeddings_llm = {}
    for concept in df_valid["cue"].unique():
        mask = df_valid["cue"] == concept
        embeddings_llm[concept] = {
            "matrix": all_vecs[mask.values],
            "run_id": df_valid[mask]["run_id"].tolist(),
        }

    if cache_file is not None:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings_llm, f)
        print(f"Saved LLM embeddings to {cache_file}.")

    return embeddings_llm


def compare_llm_human(
    embeddings_human: dict,
    embeddings_llm: dict,
    concepts: list = None,
    n_perm: int = 10000,
    seed: int = 42,
) -> dict:
    """Compare LLM-human and human-human pairwise dissimilarity distributions.

    For each concept, computes:
      hh : human-human pairwise distances (upper triangle of H x H)
      lh : LLM-human pairwise distances (full L x H matrix, flattened)
      p  : two-tailed permutation p-value on the difference in means

    The permutation test relabels which embeddings (drawn from the pooled
    set of human and LLM embeddings) are treated as "LLM" vs. "human" under
    each random permutation, preserving the dependency structure of the
    dyads (each embedding contributes to many dyads) rather than resampling
    dyad values directly, which would treat highly dependent observations
    as independent and produce an artificially narrow null distribution.

    Parameters
    ----------
    embeddings_human : dict
        Human embeddings from encode_concepts().
    embeddings_llm : dict
        LLM embeddings from encode_llm_associations().
    concepts : list of str or None
        Concepts to compare. Defaults to the intersection of both dicts.
    n_perm : int
        Number of permutation iterations.
    seed : int

    Returns
    -------
    dict
        Keys are concept strings. Values contain 'hh', 'lh' (np.ndarray),
        and 'p' (float).
    """
    if concepts is None:
        concepts = [c for c in embeddings_human if c in embeddings_llm]

    dist_data = {}

    for concept in concepts:
        if concept not in embeddings_human or concept not in embeddings_llm:
            continue

        H = embeddings_human[concept]["matrix"]
        L = embeddings_llm[concept]["matrix"]

        # Independent RNG per concept; see run_deviation_test() for rationale.
        concept_seed = (seed, concept)
        rng = np.random.default_rng(abs(hash(concept_seed)) % (2**32))

        HH_mat   = (1 - cosine_similarity(H, H)) / 2
        idx      = np.triu_indices(len(H), k=1)
        hh_dyads = HH_mat[idx]

        LH_mat   = (1 - cosine_similarity(L, H)) / 2
        lh_dyads = LH_mat.flatten()

        observed = float(lh_dyads.mean())
        mean_hh  = float(hh_dyads.mean())
        obs_diff = observed - mean_hh

        n_l, n_h   = len(L), len(H)
        n_total    = n_l + n_h
        all_emb    = np.vstack([L, H])
        full_sim   = (1 - cosine_similarity(all_emb, all_emb)) / 2
        full_idx_u = np.triu_indices(n_total, k=1)
        full_dyads = full_sim[full_idx_u]
        i_idx, j_idx = full_idx_u

        perm_diffs = np.empty(n_perm)
        for b in range(n_perm):
            perm_labels = rng.permutation(n_total) < n_l  # True = LLM
            is_cross = perm_labels[i_idx] != perm_labels[j_idx]
            is_hh    = (~perm_labels[i_idx]) & (~perm_labels[j_idx])
            perm_diffs[b] = full_dyads[is_cross].mean() - full_dyads[is_hh].mean()

        p_val = 2 * min(
            np.mean(perm_diffs >= obs_diff),
            np.mean(perm_diffs <= obs_diff)
        )

        dist_data[concept] = {"hh": hh_dyads, "lh": lh_dyads, "p": p_val}

    return dist_data


def variance_test(
    dist_data: dict,
    embeddings_llm: dict,
    n_iter: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test whether the LLM shows lower within-group spread than humans.

    Computes SD(LL) vs SD(HH) per concept and tests the difference via
    a two-tailed permutation test pooling LL and HH dyads.

    Parameters
    ----------
    dist_data : dict
        Output of compare_llm_human(). Provides 'hh' arrays.
    embeddings_llm : dict
        LLM embeddings for computing LL (LLM-LLM) distances.
    n_iter : int
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: concept, SD_HH, SD_LL, diff, p_var, sig.
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for concept, d in dist_data.items():
        if concept not in embeddings_llm:
            continue

        hh = d["hh"]
        L  = embeddings_llm[concept]["matrix"]

        LL_mat = (1 - cosine_similarity(L, L)) / 2
        ll_idx = np.triu_indices(len(L), k=1)
        ll     = LL_mat[ll_idx]

        obs_stat = ll.std() - hh.std()
        pooled   = np.concatenate([ll, hh])
        n_ll     = len(ll)

        perm_stats = np.array([
            (p := rng.permutation(pooled))[:n_ll].std() - p[n_ll:].std()
            for _ in range(n_iter)
        ])
        p_var = 2 * min(
            np.mean(perm_stats <= obs_stat),
            np.mean(perm_stats >= obs_stat)
        )

        sig = (
            "***" if p_var < 0.001 else
            "**"  if p_var < 0.01  else
            "*"   if p_var < 0.05  else
            "n.s."
        )
        rows.append({
            "concept": concept,
            "SD_HH":   hh.std(),
            "SD_LL":   ll.std(),
            "diff":    obs_stat,
            "p_var":   p_var,
            "sig":     sig,
        })

    return pd.DataFrame(rows)


def _parse_associates(raw: str):
    """Parse a comma-separated triple of associates. Returns None on failure."""
    parts = [p.strip().lower() for p in raw.strip().split(",")]
    parts = [re.sub(r"[^a-z\s\-]", "", p).strip() for p in parts]
    parts = [p for p in parts if p]
    return parts if len(parts) == 3 else None


def _build_sentence(concept: str, associates: list) -> str:
    """Construct a synthetic sentence in the same format as the SWOW pipeline."""
    return (
        f"{concept.capitalize()} is associated with "
        f"{associates[0]}, {associates[1]}, and {associates[2]}."
    )
