"""
word_properties.py
------------------
Word-level properties for correlating with sharedness (S_b).

Two sources:

1. WordNet-derived properties — always available, no extra files needed.
   Computed via nltk.corpus.wordnet for any word in the English WordNet.

2. External norms — require a separately downloaded file.
   Each function documents the expected file format and a download link.
   Only words with an available rating are retained in any analysis.

All functions return a pd.Series indexed by lowercase word string,
so they can be passed directly to correlate_with_sharedness().
"""

import numpy as np
import pandas as pd
from nltk.corpus import wordnet


# ── WordNet-derived properties ────────────────────────────────────────────────

def synset_count(words: list, pos: str = None) -> pd.Series:
    """Number of synsets (senses) in WordNet.

    A proxy for polysemy: words with more synsets are more ambiguous.
    High synset count tends to co-occur with low concreteness.

    Parameters
    ----------
    words : list of str
    pos : str or None
        Restrict to a part of speech: wordnet.NOUN, wordnet.VERB, etc.
        If None, all POS are counted.
    """
    return pd.Series(
        {w: len(wordnet.synsets(w, pos=pos)) for w in words},
        name="synset_count"
    )


def hypernym_depth(words: list, pos: str = None) -> pd.Series:
    """Mean shortest path depth from the root hypernym in the WordNet hierarchy.

    Measures taxonomic specificity: deeper words are more specific (subordinate),
    shallower words are more abstract (superordinate). Computed as the mean
    depth across all synsets for the word, using the shortest hypernym path.
    Words with no synsets for the given POS receive NaN.

    Parameters
    ----------
    words : list of str
    pos : str or None
        Part of speech. Defaults to NOUN if not specified.
    """
    pos = pos or wordnet.NOUN
    depths = {}
    for w in words:
        synsets = wordnet.synsets(w, pos=pos)
        if not synsets:
            depths[w] = np.nan
            continue
        d = [s.min_depth() for s in synsets]
        depths[w] = float(np.mean(d))
    return pd.Series(depths, name="hypernym_depth")


def hyponym_count(words: list, pos: str = None) -> pd.Series:
    """Total number of direct hyponyms (subordinate concepts) across all synsets.

    Words with many hyponyms are semantically broad (superordinate).
    Words with few hyponyms are semantically narrow (subordinate).

    Parameters
    ----------
    words : list of str
    pos : str or None
        Part of speech. Defaults to NOUN if not specified.
    """
    pos = pos or wordnet.NOUN
    counts = {}
    for w in words:
        synsets = wordnet.synsets(w, pos=pos)
        if not synsets:
            counts[w] = np.nan
            continue
        counts[w] = float(sum(len(s.hyponyms()) for s in synsets))
    return pd.Series(counts, name="hyponym_count")


def hypernym_count(words: list, pos: str = None) -> pd.Series:
    """Total number of direct hypernyms (superordinate concepts) across all synsets.

    Parameters
    ----------
    words : list of str
    pos : str or None
        Part of speech. Defaults to NOUN if not specified.
    """
    pos = pos or wordnet.NOUN
    counts = {}
    for w in words:
        synsets = wordnet.synsets(w, pos=pos)
        if not synsets:
            counts[w] = np.nan
            continue
        counts[w] = float(sum(len(s.hypernyms()) for s in synsets))
    return pd.Series(counts, name="hypernym_count")


def lemma_count(words: list, pos: str = None) -> pd.Series:
    """Total number of lemmas (surface forms) across all synsets.

    A proxy for morphological richness and lexical productivity.

    Parameters
    ----------
    words : list of str
    pos : str or None
    """
    counts = {}
    for w in words:
        synsets = wordnet.synsets(w, pos=pos)
        if not synsets:
            counts[w] = np.nan
            continue
        counts[w] = float(sum(len(s.lemmas()) for s in synsets))
    return pd.Series(counts, name="lemma_count")


def morphosemantic_links(words: list) -> pd.Series:
    """Number of morphosemantic links (derivationally related forms) across all synsets.

    Captures morphological family size. Words with many derivational links
    tend to be more central in the lexicon.

    Parameters
    ----------
    words : list of str
    """
    counts = {}
    for w in words:
        synsets = wordnet.synsets(w)
        if not synsets:
            counts[w] = np.nan
            continue
        n = sum(
            len(lemma.derivationally_related_forms())
            for s in synsets
            for lemma in s.lemmas()
        )
        counts[w] = float(n)
    return pd.Series(counts, name="morphosemantic_links")


def also_see_count(words: list) -> pd.Series:
    """Number of 'also see' relations across all synsets.

    Captures loose semantic relatedness outside the strict hypernym hierarchy.

    Parameters
    ----------
    words : list of str
    """
    counts = {}
    for w in words:
        synsets = wordnet.synsets(w)
        if not synsets:
            counts[w] = np.nan
            continue
        counts[w] = float(sum(len(s.also_sees()) for s in synsets))
    return pd.Series(counts, name="also_see_count")


def topic_domain_count(words: list) -> pd.Series:
    """Number of unique topic domain categories assigned across all synsets.

    Words assigned to many domains are semantically broad or cross-disciplinary.

    Parameters
    ----------
    words : list of str
    """
    counts = {}
    for w in words:
        synsets = wordnet.synsets(w)
        if not synsets:
            counts[w] = np.nan
            continue
        domains = set()
        for s in synsets:
            for domain in s.topic_domains():
                domains.add(domain.name())
        counts[w] = float(len(domains))
    return pd.Series(counts, name="topic_domain_count")


def pos_diversity(words: list) -> pd.Series:
    """Number of distinct parts of speech the word appears under in WordNet.

    Maximum value is 4 (noun, verb, adjective, adverb). Words that appear
    across multiple POS categories tend to be more semantically flexible.

    Parameters
    ----------
    words : list of str
    """
    counts = {}
    for w in words:
        pos_set = {s.pos() for s in wordnet.synsets(w)}
        counts[w] = float(len(pos_set)) if pos_set else np.nan
    return pd.Series(counts, name="pos_diversity")


# ── External norms ────────────────────────────────────────────────────────────
# Each function below requires a separately downloaded file.
# Only words present in the norms file are retained.

def load_concreteness(path: str) -> pd.Series:
    """Brysbaert et al. (2014) concreteness ratings.

    Download: https://link.springer.com/article/10.3758/s13423-013-0403-5
    (Supplementary material, Excel file)
    Expected columns: 'Word', 'Conc.M'

    Scale: 1 (abstract) to 5 (concrete).
    """
    df = pd.read_excel(path)
    return df.set_index(df['Word'].str.lower())['Conc.M'].rename("concreteness")


def load_valence(path: str) -> pd.Series:
    """Warriner et al. (2013) affective norms: valence.

    Download: https://link.springer.com/article/10.3758/s13428-012-0314-x
    (Supplementary material, CSV file)
    Expected columns: 'Word', 'V.Mean.Sum'

    Scale: 1 (negative) to 9 (positive).
    """
    df = pd.read_csv(path)
    return df.set_index(df['Word'].str.lower())['V.Mean.Sum'].rename("valence")


def load_arousal(path: str) -> pd.Series:
    """Warriner et al. (2013) affective norms: arousal.

    Download: https://link.springer.com/article/10.3758/s13428-012-0314-x
    Expected columns: 'Word', 'A.Mean.Sum'

    Scale: 1 (calm) to 9 (arousing).
    """
    df = pd.read_csv(path)
    return df.set_index(df['Word'].str.lower())['A.Mean.Sum'].rename("arousal")


def load_dominance(path: str) -> pd.Series:
    """Warriner et al. (2013) affective norms: dominance.

    Download: https://link.springer.com/article/10.3758/s13428-012-0314-x
    Expected columns: 'Word', 'D.Mean.Sum'

    Scale: 1 (submissive) to 9 (dominant).
    """
    df = pd.read_csv(path)
    return df.set_index(df['Word'].str.lower())['D.Mean.Sum'].rename("dominance")


def load_age_of_acquisition(path: str) -> pd.Series:
    """Kuperman et al. (2012) age-of-acquisition norms.

    Download: https://link.springer.com/article/10.3758/s13428-012-0210-4
    (Supplementary material, Excel file)
    Expected columns: 'Word', 'Rating.Mean'

    Scale: estimated age in years at which the word was learned.
    """
    df = pd.read_excel(path)
    return df.set_index(df['Word'].str.lower())['Rating.Mean'].rename("age_of_acquisition")


def load_word_frequency(path: str) -> pd.Series:
    """Brysbaert & New (2009) SUBTLEX-US word frequency norms.

    Download: https://www.ugent.be/pp/experimentele-psychologie/en/research/psycholinguistics/subtlexus
    Expected columns: 'Word', 'SUBTLWF' (frequency per million words)

    Log-transform recommended before use: np.log1p(series)
    """
    df = pd.read_excel(path)
    return df.set_index(df['Word'].str.lower())['SUBTLWF'].rename("word_frequency")


def load_imageability(path: str) -> pd.Series:
    """Cortese & Fugett (2004) or Gilhooly & Logie (1980) imageability norms.

    No single canonical download; Cortese & Fugett available via:
    https://link.springer.com/article/10.3758/BF03195585
    Expected columns: 'Word', 'Imageability'

    Scale varies by source; check documentation.
    """
    df = pd.read_csv(path)
    return df.set_index(df['Word'].str.lower())['Imageability'].rename("imageability")


# ── Utilities ─────────────────────────────────────────────────────────────────

def build_properties_table(words: list, properties: dict) -> pd.DataFrame:
    """Combine multiple word-level properties into a single DataFrame.

    Words are retained only where ALL requested properties have a value.
    Words missing from any property are dropped with a printed summary.

    Parameters
    ----------
    words : list of str
        The concept set to look up.
    properties : dict
        Mapping from property name (str) to pd.Series indexed by word.
        Use the functions above to construct each Series, e.g.:
            {
                'concreteness':  load_concreteness('word_ratings.xlsx'),
                'hypernym_depth': hypernym_depth(words),
                'synset_count':  synset_count(words),
            }

    Returns
    -------
    pd.DataFrame
        Index: word. Columns: one per property. Only complete rows retained.

    Example
    -------
    >>> props = build_properties_table(
    ...     words=['rain', 'justice', 'tree'],
    ...     properties={
    ...         'concreteness':  load_concreteness('word_ratings.xlsx'),
    ...         'hypernym_depth': hypernym_depth(['rain', 'justice', 'tree']),
    ...     }
    ... )
    """
    df = pd.DataFrame(index=[w.lower() for w in words])

    for name, series in properties.items():
        series.index = series.index.str.lower()
        df[name] = series.reindex(df.index)

    n_before = len(df)
    df = df.dropna()
    n_after  = len(df)

    if n_before > n_after:
        print(f"Dropped {n_before - n_after} concepts with missing values "
              f"({n_after} retained).")

    return df
