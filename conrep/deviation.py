"""
deviation.py
------------
Avenue 3 — Examining Deviation.

Tests whether a target group's concept representations deviate from a
human reference distribution. The test compares cross-group pairwise
distances (target-to-reference) against within-reference distances via
a permutation test on the mean.

Two subsections:

A) Subpopulation within the dataset
   Build a target group from participant metadata. For SWOW data, a set
   of pre-built subgroups is available. For any dataset, a custom filter
   expression can be used.

B) Subpopulation from external data
   Encode associations from any external source (e.g. an LLM) using the
   same pipeline as human participants, then run the same test.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# UN large-city threshold for urban/rural classification.
# Source: United Nations, World Urbanization Prospects.
URBAN_POPULATION_THRESHOLD = 300_000


# ── Subsection A: subpopulation from within dataset ───────────────────────────

def build_swow_subgroups(
    df: pd.DataFrame,
    df_participants: pd.DataFrame = None,
    participant_col: str = 'participantID',
) -> dict:
    """Return a dict of pre-built SWOW subgroup participant ID sets.

    Each value is a set of participantIDs belonging to the subgroup.
    Subgroups are mutually exclusive within their category but not across
    categories.

    Requires df to have columns: age, gender, nativeLanguage.
    Requires df_participants (output of build_participant_geo in geo.py)
    for location-based subgroups (urban/rural, continent).

    Parameters
    ----------
    df : pd.DataFrame
        Filtered association DataFrame.
    df_participants : pd.DataFrame or None
        Participant geo table from geo.py. Required for urban/rural and
        continent subgroups.
    participant_col : str

    Returns
    -------
    dict
        Keys are subgroup names. Values are sets of participantIDs.
    """
    df_p = df[[participant_col] + [
        c for c in ['age', 'gender', 'nativeLanguage', 'education']
        if c in df.columns
    ]].drop_duplicates(subset=participant_col).copy()

    subgroups = {}

    # Age cohorts
    if 'age' in df_p.columns:
        df_p['age_num'] = pd.to_numeric(df_p['age'], errors='coerce')
        subgroups['under_25']  = set(df_p[df_p['age_num'] < 25][participant_col])
        subgroups['above_65']  = set(df_p[df_p['age_num'] > 65][participant_col])

    # Gender — SWOW-EN2018 encodes this as 'Ma' (male), 'Fe' (female), 'X' (other/undisclosed)
    if 'gender' in df_p.columns:
        g = df_p['gender'].astype(str).str.strip().str.lower()
        subgroups['women'] = set(df_p[g.isin(['fe', 'f', 'female', 'woman'])][participant_col])
        subgroups['men']   = set(df_p[g.isin(['ma', 'm', 'male', 'man'])][participant_col])

    # Native language — SWOW-EN2018's nativeLanguage field is actually a mix of
    # countries and language names. English-speaking entries are identified by
    # country/region rather than a literal 'english' value.
    if 'nativeLanguage' in df_p.columns:
        english_speaking = {
            'united states', 'united kingdom', 'australia', 'canada',
            'new zealand', 'ireland', 'other_english',
        }
        nl = df_p['nativeLanguage'].astype(str).str.strip().str.lower()
        subgroups['native_english']     = set(df_p[nl.isin(english_speaking)][participant_col])
        subgroups['non_native_english'] = set(df_p[~nl.isin(english_speaking)][participant_col])

    # Education (available for subset of SWOW)
    if 'education' in df_p.columns:
        edu = pd.to_numeric(df_p['education'], errors='coerce')
        # 4 = Bachelor, 5 = Master (see SWOW codebook)
        subgroups['university_degree'] = set(df_p[edu >= 4][participant_col])

    # Location-based (requires df_participants from geo.py)
    if df_participants is not None and 'city_population' in df_participants.columns:
        urban_ids = set(df_participants[
            df_participants['city_population'] >= URBAN_POPULATION_THRESHOLD
        ][participant_col])
        rural_ids = set(df_participants[
            df_participants['city_population'] < URBAN_POPULATION_THRESHOLD
        ][participant_col])
        subgroups['urban'] = urban_ids   # city population >= 300,000 (UN threshold)
        subgroups['rural'] = rural_ids

    # Continent-level subgroups (requires df_participants with country_code)
    if df_participants is not None and 'country_code' in df_participants.columns:
        continent_map = _country_to_continent()
        df_participants = df_participants.copy()
        df_participants['continent'] = df_participants['country_code'].map(continent_map)
        for continent in ['Europe', 'Africa', 'Asia', 'Americas', 'Oceania']:
            ids = set(df_participants[
                df_participants['continent'] == continent
            ][participant_col])
            if ids:
                subgroups[continent.lower()] = ids

    # Print summary
    print(f"Available SWOW subgroups ({len(subgroups)}):")
    for name, ids in sorted(subgroups.items(), key=lambda x: -len(x[1])):
        print(f"  {name:<25} N = {len(ids)}")

    return subgroups


def build_custom_subgroup(
    df: pd.DataFrame,
    filter_expr: str,
    participant_col: str = 'participantID',
) -> set:
    """Build a subgroup by evaluating a filter expression on df.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered association DataFrame.
    filter_expr : str
        A pandas query string, e.g. "age > 65" or "gender == 'f'".
    participant_col : str

    Returns
    -------
    set of participantIDs
    """
    df_p = df.drop_duplicates(subset=participant_col)
    result = set(df_p.query(filter_expr)[participant_col])
    print(f"Custom subgroup '{filter_expr}': N = {len(result)}")
    return result


def run_deviation_test(
    embeddings: dict,
    target_ids: set,
    concepts: list,
    n_perm: int = 10000,
    seed: int = 42,
    min_n: int = 75,
    warn_n: int = 20,
    participant_col: str = 'participantID',
    return_distributions: bool = False,
):
    """Compare target group to reference distribution per concept.

    For each concept, computes:
      mean_cross : mean pairwise distance between target and reference participants
      mean_ref   : mean pairwise distance within the reference group
      p          : two-tailed permutation p-value on the difference in means

    The reference group is all participants NOT in the target group.
    Concepts are dropped only if N_target + N_reference < min_n (the same
    threshold used for MIN_PARTICIPANTS in Section 1 — i.e. concepts that
    didn't have enough total respondents to begin with). A concept is never
    dropped for having a small target group on its own; instead, a warning
    is printed if N_target < warn_n, and the concept is still included in
    the output, flagged.

    Parameters
    ----------
    embeddings : dict
        Concept-keyed embeddings dict from encode_concepts().
    target_ids : set
        participantIDs of the target group.
    concepts : list of str
    n_perm : int
    seed : int
    min_n : int
        Minimum combined N (N_target + N_reference) required per concept.
    warn_n : int
        Print a warning if N_target falls below this threshold.
    participant_col : str
    return_distributions : bool
        If True, also return a dist_data dict (concept -> {'hh', 'lh', 'p'})
        in the same format as compare_llm_human(), ready to pass directly
        to plot_llm_distributions(). 'hh' here is the within-reference
        distribution and 'lh' is the cross (target-reference) distribution.

    Returns
    -------
    pd.DataFrame
        Columns: concept, N_target, N_reference, mean_cross, mean_ref, p, sig.
        Sorted by p ascending.
    dict, only if return_distributions=True
        Concept-keyed dict with 'hh' (within-reference dyads), 'lh' (cross
        dyads), and 'p' (the same permutation p-value as in the DataFrame).
    """
    rows = []
    dist_data = {}

    for concept in concepts:
        if concept not in embeddings:
            continue

        pids   = embeddings[concept]['participantID']
        matrix = embeddings[concept]['matrix']

        target_idx = [i for i, p in enumerate(pids) if p in target_ids]
        ref_idx    = [i for i, p in enumerate(pids) if p not in target_ids]

        n_t = len(target_idx)
        n_r = len(ref_idx)

        if (n_t + n_r) < min_n:
            continue

        if n_t < warn_n:
            print(f"Warning [{concept}]: N_target={n_t} — low N")

        # Each concept gets its own independent RNG, seeded deterministically
        # from the base seed and the concept name. This guarantees the
        # permutation distribution for one concept can never be affected by
        # how many random draws were consumed processing earlier concepts in
        # the same call, while remaining fully reproducible.
        concept_seed = (seed, concept)
        rng = np.random.default_rng(abs(hash(concept_seed)) % (2**32))

        T = matrix[target_idx]
        R = matrix[ref_idx]

        cross_mat = (1 - cosine_similarity(T, R)) / 2
        ref_mat   = (1 - cosine_similarity(R, R)) / 2
        ref_idx_u = np.triu_indices(n_r, k=1)

        cross_dyads = cross_mat.flatten()
        ref_dyads   = ref_mat[ref_idx_u]

        mean_cross = float(cross_dyads.mean())
        mean_ref   = float(ref_dyads.mean())
        obs_diff   = mean_cross - mean_ref

        # Permutation test: shuffle which participants are labeled "target"
        # vs. "reference" among the full pool of n_t + n_r participants, then
        # recompute the same statistic under each random relabeling. This
        # preserves the true dependency structure (each participant
        # contributes to multiple dyads) and the correct effective sample
        # size, unlike resampling dyads directly with replacement, which
        # treats highly dependent dyad values as if they were independent
        # observations and produces an artificially narrow null distribution.
        all_emb = np.vstack([T, R])
        n_total = n_t + n_r
        full_sim = (1 - cosine_similarity(all_emb, all_emb)) / 2
        full_idx_u = np.triu_indices(n_total, k=1)
        full_dyads = full_sim[full_idx_u]
        i_idx, j_idx = full_idx_u

        perm_diffs = np.empty(n_perm)
        for b in range(n_perm):
            perm_labels = rng.permutation(n_total) < n_t  # True = target
            is_cross = perm_labels[i_idx] != perm_labels[j_idx]
            is_ref   = (~perm_labels[i_idx]) & (~perm_labels[j_idx])
            perm_mean_cross = full_dyads[is_cross].mean()
            perm_mean_ref   = full_dyads[is_ref].mean()
            perm_diffs[b] = perm_mean_cross - perm_mean_ref

        p_val = 2 * min(
            np.mean(perm_diffs >= obs_diff),
            np.mean(perm_diffs <= obs_diff)
        )

        rows.append({
            'concept':     concept,
            'N_target':    n_t,
            'N_reference': n_r,
            'mean_cross':  round(mean_cross, 4),
            'mean_ref':    round(mean_ref, 4),
            'p':           round(p_val, 4),
            'sig':         _sig_stars(p_val),
        })

        if return_distributions:
            dist_data[concept] = {'hh': ref_dyads, 'lh': cross_dyads, 'p': p_val}

    if len(rows) == 0:
        print(
            f"No concepts had a combined N_target + N_reference >= {min_n}. "
            f"Check that target_ids actually match participantIDs in your data, "
            f"or choose concepts with more total respondents."
        )
        df_result = pd.DataFrame(columns=[
            'concept', 'N_target', 'N_reference', 'mean_cross', 'mean_ref', 'p', 'sig'
        ])
    else:
        df_result = (
            pd.DataFrame(rows)
            .sort_values('p')
            .reset_index(drop=True)
        )

    if return_distributions:
        return df_result, dist_data
    return df_result


# ── Subsection B: subpopulation from external data ────────────────────────────

def run_deviation_test_external(
    embeddings_reference: dict,
    embeddings_target: dict,
    concepts: list,
    n_perm: int = 10000,
    seed: int = 42,
    warn_n: int = 20,
) -> pd.DataFrame:
    """Compare an external target group to the human reference distribution.

    Same test as run_deviation_test() but takes two separate embeddings dicts
    rather than splitting one by participant IDs. Use this for LLMs or any
    externally collected associations.

    Parameters
    ----------
    embeddings_reference : dict
        Human embeddings from encode_concepts().
    embeddings_target : dict
        Target group embeddings (e.g. from encode_llm_associations()).
    concepts : list of str
    n_perm, seed, warn_n : as in run_deviation_test()

    Returns
    -------
    pd.DataFrame
        Same columns as run_deviation_test().
    """
    rows = []

    for concept in concepts:
        if concept not in embeddings_reference or concept not in embeddings_target:
            continue

        R = embeddings_reference[concept]['matrix']
        T = embeddings_target[concept]['matrix']

        n_r = len(R)
        n_t = len(T)

        if n_t < warn_n:
            print(f"Warning [{concept}]: N_target={n_t} — low N")

        # Independent RNG per concept; see run_deviation_test() for rationale.
        concept_seed = (seed, concept)
        rng = np.random.default_rng(abs(hash(concept_seed)) % (2**32))

        cross_mat = (1 - cosine_similarity(T, R)) / 2
        ref_mat   = (1 - cosine_similarity(R, R)) / 2
        ref_idx_u = np.triu_indices(n_r, k=1)

        mean_cross = float(cross_mat.mean())
        mean_ref   = float(ref_mat[ref_idx_u].mean())
        obs_diff   = mean_cross - mean_ref

        # Permutation test: relabel which embeddings belong to "target" vs.
        # "reference" among the pooled set, preserving dependency structure
        # rather than treating dyads as independent (see run_deviation_test
        # for the full rationale).
        all_emb    = np.vstack([T, R])
        n_total    = n_t + n_r
        full_sim   = (1 - cosine_similarity(all_emb, all_emb)) / 2
        full_idx_u = np.triu_indices(n_total, k=1)
        full_dyads = full_sim[full_idx_u]
        i_idx, j_idx = full_idx_u

        perm_diffs = np.empty(n_perm)
        for b in range(n_perm):
            perm_labels = rng.permutation(n_total) < n_t
            is_cross = perm_labels[i_idx] != perm_labels[j_idx]
            is_ref   = (~perm_labels[i_idx]) & (~perm_labels[j_idx])
            perm_diffs[b] = full_dyads[is_cross].mean() - full_dyads[is_ref].mean()

        p_val = 2 * min(
            np.mean(perm_diffs >= obs_diff),
            np.mean(perm_diffs <= obs_diff)
        )

        rows.append({
            'concept':     concept,
            'N_target':    n_t,
            'N_reference': n_r,
            'mean_cross':  round(mean_cross, 4),
            'mean_ref':    round(mean_ref, 4),
            'p':           round(p_val, 4),
            'sig':         _sig_stars(p_val),
        })

    if len(rows) == 0:
        print(
            f"No concepts met the warn_n >= {warn_n} threshold for both target "
            f"and reference group, or no concepts were found in both embeddings dicts."
        )
        return pd.DataFrame(columns=[
            'concept', 'N_target', 'N_reference', 'mean_cross', 'mean_ref', 'p', 'sig'
        ])

    return (
        pd.DataFrame(rows)
        .sort_values('p')
        .reset_index(drop=True)
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _country_to_continent() -> dict:
    """ISO-3166 alpha-2 -> continent name mapping."""
    return {
        'AF': 'Asia', 'AL': 'Europe', 'DZ': 'Africa', 'AD': 'Europe',
        'AO': 'Africa', 'AG': 'Americas', 'AR': 'Americas', 'AM': 'Asia',
        'AU': 'Oceania', 'AT': 'Europe', 'AZ': 'Asia', 'BS': 'Americas',
        'BH': 'Asia', 'BD': 'Asia', 'BB': 'Americas', 'BY': 'Europe',
        'BE': 'Europe', 'BZ': 'Americas', 'BJ': 'Africa', 'BT': 'Asia',
        'BO': 'Americas', 'BA': 'Europe', 'BW': 'Africa', 'BR': 'Americas',
        'BN': 'Asia', 'BG': 'Europe', 'BF': 'Africa', 'BI': 'Africa',
        'CV': 'Africa', 'KH': 'Asia', 'CM': 'Africa', 'CA': 'Americas',
        'CF': 'Africa', 'TD': 'Africa', 'CL': 'Americas', 'CN': 'Asia',
        'CO': 'Americas', 'KM': 'Africa', 'CD': 'Africa', 'CG': 'Africa',
        'CR': 'Americas', 'HR': 'Europe', 'CU': 'Americas', 'CY': 'Asia',
        'CZ': 'Europe', 'DK': 'Europe', 'DJ': 'Africa', 'DM': 'Americas',
        'DO': 'Americas', 'EC': 'Americas', 'EG': 'Africa', 'SV': 'Americas',
        'GQ': 'Africa', 'ER': 'Africa', 'EE': 'Europe', 'SZ': 'Africa',
        'ET': 'Africa', 'FJ': 'Oceania', 'FI': 'Europe', 'FR': 'Europe',
        'GA': 'Africa', 'GM': 'Africa', 'GE': 'Asia', 'DE': 'Europe',
        'GH': 'Africa', 'GR': 'Europe', 'GD': 'Americas', 'GT': 'Americas',
        'GN': 'Africa', 'GW': 'Africa', 'GY': 'Americas', 'HT': 'Americas',
        'HN': 'Americas', 'HU': 'Europe', 'IS': 'Europe', 'IN': 'Asia',
        'ID': 'Asia', 'IR': 'Asia', 'IQ': 'Asia', 'IE': 'Europe',
        'IL': 'Asia', 'IT': 'Europe', 'JM': 'Americas', 'JP': 'Asia',
        'JO': 'Asia', 'KZ': 'Asia', 'KE': 'Africa', 'KI': 'Oceania',
        'KP': 'Asia', 'KR': 'Asia', 'KW': 'Asia', 'KG': 'Asia',
        'LA': 'Asia', 'LV': 'Europe', 'LB': 'Asia', 'LS': 'Africa',
        'LR': 'Africa', 'LY': 'Africa', 'LI': 'Europe', 'LT': 'Europe',
        'LU': 'Europe', 'MG': 'Africa', 'MW': 'Africa', 'MY': 'Asia',
        'MV': 'Asia', 'ML': 'Africa', 'MT': 'Europe', 'MH': 'Oceania',
        'MR': 'Africa', 'MU': 'Africa', 'MX': 'Americas', 'FM': 'Oceania',
        'MD': 'Europe', 'MC': 'Europe', 'MN': 'Asia', 'ME': 'Europe',
        'MA': 'Africa', 'MZ': 'Africa', 'MM': 'Asia', 'NA': 'Africa',
        'NR': 'Oceania', 'NP': 'Asia', 'NL': 'Europe', 'NZ': 'Oceania',
        'NI': 'Americas', 'NE': 'Africa', 'NG': 'Africa', 'MK': 'Europe',
        'NO': 'Europe', 'OM': 'Asia', 'PK': 'Asia', 'PW': 'Oceania',
        'PA': 'Americas', 'PG': 'Oceania', 'PY': 'Americas', 'PE': 'Americas',
        'PH': 'Asia', 'PL': 'Europe', 'PT': 'Europe', 'QA': 'Asia',
        'RO': 'Europe', 'RU': 'Europe', 'RW': 'Africa', 'KN': 'Americas',
        'LC': 'Americas', 'VC': 'Americas', 'WS': 'Oceania', 'SM': 'Europe',
        'ST': 'Africa', 'SA': 'Asia', 'SN': 'Africa', 'RS': 'Europe',
        'SC': 'Africa', 'SL': 'Africa', 'SG': 'Asia', 'SK': 'Europe',
        'SI': 'Europe', 'SB': 'Oceania', 'SO': 'Africa', 'ZA': 'Africa',
        'SS': 'Africa', 'ES': 'Europe', 'LK': 'Asia', 'SD': 'Africa',
        'SR': 'Americas', 'SE': 'Europe', 'CH': 'Europe', 'SY': 'Asia',
        'TW': 'Asia', 'TJ': 'Asia', 'TZ': 'Africa', 'TH': 'Asia',
        'TL': 'Asia', 'TG': 'Africa', 'TO': 'Oceania', 'TT': 'Americas',
        'TN': 'Africa', 'TR': 'Asia', 'TM': 'Asia', 'TV': 'Oceania',
        'UG': 'Africa', 'UA': 'Europe', 'AE': 'Asia', 'GB': 'Europe',
        'US': 'Americas', 'UY': 'Americas', 'UZ': 'Asia', 'VU': 'Oceania',
        'VE': 'Americas', 'VN': 'Asia', 'YE': 'Asia', 'ZM': 'Africa',
        'ZW': 'Africa',
    }
