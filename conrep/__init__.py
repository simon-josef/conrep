"""
conrep
-------
A framework for measuring sharedness and idiosyncrasy in concept representations
using sentence embeddings and pairwise distance modeling.

Public API
----------
Word properties
    synset_count, hypernym_depth, hyponym_count, hypernym_count,
    lemma_count, morphosemantic_links, also_see_count, topic_domain_count,
    pos_diversity, load_concreteness, load_valence, load_arousal,
    load_dominance, load_age_of_acquisition, load_word_frequency,
    load_imageability, build_properties_table

Preprocessing
    prepare_swow        Filter, construct sentences, ready for encoding.
    prepare_custom      Validate a bring-your-own-sentences DataFrame.

Embeddings
    load_encoder, encode_concepts, subset_embeddings

Sharedness
    compute_sharedness, compute_pairwise_distances

Mantel tests
    build_dyads, mantel_test

Machine behavior
    collect_llm_associations, encode_llm_associations,
    compare_llm_human, variance_test

Plotting
    set_style, plot_sharedness, plot_partial_regression,
    plot_mantel_results, plot_mantel_joint, plot_distribution_comparison,
    plot_llm_distributions
"""

from .word_properties import (
    synset_count,
    hypernym_depth,
    hyponym_count,
    hypernym_count,
    lemma_count,
    morphosemantic_links,
    also_see_count,
    topic_domain_count,
    pos_diversity,
    load_concreteness,
    load_valence,
    load_arousal,
    load_dominance,
    load_age_of_acquisition,
    load_word_frequency,
    load_imageability,
    build_properties_table,
)

from .preprocessing import (
    prepare_swow,
    prepare_custom,
    build_sentence,
    is_valid_response,
    has_noun_synset,
    clean_response,
)

from .embeddings import (
    load_encoder,
    encode_concepts,
    subset_embeddings,
)

from .sharedness import (
    compute_sharedness,
    compute_pairwise_distances,
)

from .mantel import (
    build_dyads,
    mantel_test,
)

from .llm import (
    collect_llm_associations,
    encode_llm_associations,
    compare_llm_human,
    variance_test,
)

from .plotting import (
    set_style,
    plot_sharedness,
    plot_partial_regression,
    plot_mantel_results,
    plot_mantel_joint,
    plot_distribution_comparison,
    plot_llm_distributions,
)

__version__ = "0.1.0"
__author__  = "Simon Josef Durstewitz"

from .geo import load_swow_predictors, GeoCoordinates

from .deviation import (
    build_swow_subgroups,
    build_custom_subgroup,
    run_deviation_test,
    run_deviation_test_external,
    URBAN_POPULATION_THRESHOLD,
)
