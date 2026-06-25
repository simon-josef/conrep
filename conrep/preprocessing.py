"""
preprocessing.py
----------------
Two data ingestion modes:

1. SWOW format (or any fixed-response association dataset)
   Rows contain a cue and a fixed number of response columns (R1, R2, ..., Rn).
   Synthetic sentences are constructed from these responses and passed to the
   encoder. All rows must have exactly MIN_RESPONSES valid (non-missing,
   non-sentinel) responses; rows with fewer are dropped.

   Sentence template (editable):
       "{Cue} is associated with R1, R2, and R3."

2. Custom sentences
   Researcher supplies a DataFrame with columns participantID, cue, sentence.
   No preprocessing is performed; the sentence column is passed directly to
   the encoder.
"""

import re
import string

import numpy as np
import pandas as pd
from nltk.corpus import wordnet


SWOW_MISSING = {"NA", "No more responses", "Unknown word"}


def is_valid_response(x) -> bool:
    """Return True iff x is a non-empty string that is not a SWOW sentinel value."""
    if not isinstance(x, str):
        return False
    s = x.strip()
    return len(s) > 0 and s not in SWOW_MISSING


def has_noun_synset(word) -> bool:
    """Return True iff word has at least one nominal synset in WordNet."""
    if not isinstance(word, str):
        return False
    return len(wordnet.synsets(word, pos=wordnet.NOUN)) > 0


def clean_response(x) -> str:
    """Clean a single response: lowercase, strip, remove punctuation.

    Returns the cleaned string, or None if the input is missing, a SWOW
    sentinel value, or empty after cleaning (e.g. punctuation-only responses
    such as "..." or "-").
    """
    if not isinstance(x, str):
        return None
    s = x.strip()
    if len(s) == 0 or s in SWOW_MISSING:
        return None
    cleaned = s.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    return cleaned if len(cleaned) > 0 else None


def _format_list(items: list) -> str:
    """Format a list of strings as a natural English enumeration.

    Examples
    --------
    ['wet']              -> 'wet'
    ['wet', 'cold']      -> 'wet and cold'
    ['wet', 'cold', 'grey'] -> 'wet, cold, and grey'
    """
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_sentence(
    cue: str,
    responses: list,
    template: str = "{cue} is associated with {responses}.",
) -> str:
    """Construct a synthetic sentence from a cue and its responses.

    Parameters
    ----------
    cue : str
        The concept word.
    responses : list of str
        Valid, cleaned response strings.
    template : str
        Sentence template. Must contain {cue} and {responses} placeholders.
        Default: "{cue} is associated with {responses}."

    Returns
    -------
    str
    """
    return template.format(
        cue=cue.capitalize(),
        responses=_format_list(responses),
    )


def prepare_swow(
    df: pd.DataFrame,
    response_cols: list,
    min_responses: int,
    cue_col: str = "cue",
    participant_col: str = "participantID",
    noun_filter: bool = True,
    min_participants: int = 75,
    template: str = "{cue} is associated with {responses}.",
) -> pd.DataFrame:
    """Prepare a SWOW-format DataFrame for encoding.

    Filters rows, constructs synthetic sentences, and returns a clean
    DataFrame ready for encode_concepts().

    Parameters
    ----------
    df : pd.DataFrame
        Raw association DataFrame.
    response_cols : list of str
        Column names for the response slots, e.g. ['R1', 'R2', 'R3'].
        Can be any length; the same number of valid responses is required
        in every retained row.
    min_responses : int
        Minimum number of valid (non-missing, non-sentinel) responses
        required per row. Rows with fewer are dropped.
        Must be <= len(response_cols).
    cue_col : str
        Column containing the concept label.
    participant_col : str
        Column containing participant identifiers.
    noun_filter : bool
        Whether to restrict cues to those with a WordNet noun synset.
        Set to False if your concept set is pre-selected.
    min_participants : int
        Minimum number of valid rows required per cue after all filters.
    template : str
        Sentence template. Must contain {cue} and {responses} placeholders.

    Returns
    -------
    pd.DataFrame
        Columns: cue_col, participant_col, response_cols..., sentence.
        One row per participant per cue.
    """
    assert min_responses <= len(response_cols), (
        f"min_responses ({min_responses}) cannot exceed "
        f"the number of response columns ({len(response_cols)})."
    )

    df = df.copy()

    # Noun filter
    if noun_filter:
        noun_cues = {c for c in df[cue_col].unique() if has_noun_synset(c)}
        df = df[df[cue_col].isin(noun_cues)].copy()

    # Clean every response column. Invalid responses (missing, sentinel,
    # or punctuation-only) become None.
    for col in response_cols:
        df[col] = df[col].apply(clean_response)

    # A row is kept only if it has at least min_responses valid responses.
    n_valid = df[response_cols].notna().sum(axis=1)
    df = df[n_valid >= min_responses].copy()

    # Keep only cues with at least min_participants valid rows.
    counts = df[cue_col].value_counts()
    df = df[df[cue_col].isin(counts[counts >= min_participants].index)].copy()

    # Build the synthetic sentence from the valid responses in each row.
    def _build(row):
        responses = [row[c] for c in response_cols if pd.notna(row[c])]
        return build_sentence(row[cue_col], responses, template)

    df["sentence"] = df.apply(_build, axis=1)

    print(f"Cues retained  : {df[cue_col].nunique()}")
    print(f"Rows retained  : {len(df)}")

    return df.reset_index(drop=True)


def prepare_custom(
    df: pd.DataFrame,
    cue_col: str = "cue",
    participant_col: str = "participantID",
    sentence_col: str = "sentence",
) -> pd.DataFrame:
    """Validate and return a custom sentences DataFrame.

    No sentence construction is performed. The DataFrame must already
    contain one sentence per participant per cue.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain cue_col, participant_col, sentence_col.
    cue_col, participant_col, sentence_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Validated copy with only the three required columns retained
        (plus any others present).
    """
    required = {cue_col, participant_col, sentence_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.dropna(subset=[sentence_col]).copy()

    print(f"Cues retained  : {df[cue_col].nunique()}")
    print(f"Rows retained  : {len(df)}")

    return df.reset_index(drop=True)
