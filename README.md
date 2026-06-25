# conrep

Code accompanying:

> Durstewitz, S. J. (2026). *Concept Representations in Vector Space: Avenues for the Interdisciplinary Study of Meaning Variation.*

Concept representations are the internal structures through which individuals interpret and navigate the world. **conrep** provides a measurement framework for studying interindividual variation in concept representations: short textual responses elicited for concepts are encoded as sentence embeddings, placing individual responses in a shared vector space from which pairwise representational distances are modeled.

---

## Quick Start

```bash
git clone https://github.com/simon-josef/conrep
cd conrep
pip install -r requirements.txt
```

Open `analysis.ipynb` and run top to bottom. Configuration cells throughout the notebook let you set data paths, concept sets, predictors, and target groups for each analytical avenue — the framework is not tied to any fixed concept set, predictor, or model.

---

## The Measurement Framework

For each concept $b$, every participant's response is encoded as a sentence embedding $v_{ib}$. For every pair of participants $i, j$, the cosine dissimilarity is computed as:

$$\delta_{ijb} = \frac{1 - v_i^\top v_j}{2} \in [0,1]$$

yielding $N(N-1)/2$ pairwise dissimilarities per concept. These dyadic distances are the point of departure for three analytical avenues:

### 1. Comparing Sharedness

Quantifies how shared or fragile a concept's representation is across individuals. Sharedness is defined as the complement of the mean pairwise dissimilarity:

$$S_b = 1 - \bar{\delta}_b, \quad \bar{\delta}_b = \frac{2}{N(N-1)} \sum_{i<j} \delta_{ijb}$$

High $S_b$ indicates greater representational agreement across individuals. Sharedness can be compared across concepts and correlated with concept-level properties (e.g. concreteness, taxonomic depth, polysemy).

### 2. Locating Variation

Identifies variables that systematically pattern representational variation. For a participant-level predictor $p_i$, the pairwise predictor distance is $x_{ij} = |p_i - p_j|$. A Mantel test assesses the correlation between $\delta_{ijb}$ and $x_{ij}$ across all dyads for a concept, testing whether participants who differ more on the predictor also differ more in their representation of the concept.

### 3. Examining Deviation

Positions a focal actor or subpopulation $A$ relative to a reference population $R$. Two pairwise dissimilarity distributions are constructed per concept: the within-reference distribution $\delta_{RR,b}$ and the cross-group distribution $\delta_{AR,b}$ (every member of $A$ against every member of $R$). Comparing the two locates the focal actor or group relative to typical interindividual variation, with significance assessed via permutation test.

---

## Repository Structure

```
conrep/
├── analysis.ipynb           ← configure and run here
├── requirements.txt
├── README.md
└── conrep/
    ├── __init__.py          ← public API
    ├── preprocessing.py     ← sentence construction and response filtering
    ├── embeddings.py        ← encoding and disk caching
    ├── word_properties.py   ← WordNet-derived and external concept-level properties
    ├── sharedness.py        ← Avenue 1: Comparing Sharedness
    ├── geo.py                ← participant-level predictors from SWOW metadata
    ├── mantel.py             ← Avenue 2: Locating Variation
    ├── deviation.py          ← Avenue 3: Examining Deviation
    ├── llm.py                ← LLM elicitation and comparison (multi-provider: Gemini, OpenAI, Anthropic)
    └── plotting.py           ← all figure functions
```

---

## Data

The notebook accepts any dataset structured as one row per participant per concept, or pre-built sentences passed directly to the encoder. It is demonstrated on the Small World of Words English free-association norms (SWOW-EN2018; De Deyne et al., 2019): 83,864 participants, 12,218 cue words, 3,684,600 responses, with self-reported age, native language, education, gender, and city/country of residence.

| Resource | Source |
|---|---|
| SWOW-EN2018 | [smallworldofwords.org](https://smallworldofwords.org/project/research/) — CC BY-NC-ND 3.0 |
| Concreteness / valence / arousal / dominance / age-of-acquisition / frequency norms (optional) | see `word_properties.py` docstrings for sources and download links |

GeoNames `cities500.txt` and World Bank country indicators (population, GDP per capita) are fetched automatically on first run via `load_swow_predictors()`.

---

## Using Your Own Data

Any dataset structured as `(participantID, cue, R1, R2, ...)` rows works with `prepare_swow()`. Pass `noun_filter=False` if your concept set is pre-selected. Pre-built sentences can be passed directly via `prepare_custom()`.

To use a custom participant-level predictor (Avenue 2):

```python
my_predictor = my_dataframe.set_index('participantID')['education_years']
df_dyads = fw.build_dyads(embeddings, predictor=my_predictor, concepts=MY_CONCEPTS)
results  = fw.mantel_test(df_dyads)
```

To compare a custom target group against the reference distribution (Avenue 3):

```python
target_ids = fw.build_custom_subgroup(df, "age > 65", participant_col='participantID')
results, dist_data = fw.run_deviation_test(embeddings, target_ids=target_ids, concepts=MY_CONCEPTS, return_distributions=True)
```

The framework is agnostic to elicitation method, concept set, encoder, and predictor — it can be applied to free association, definitions, narratives, interviews, or any other form of open-ended linguistic production.

---

## Citation

```bibtex
@misc{durstewitz2026conrep,
  author = {Durstewitz, Simon Josef},
  title  = {Concept Representations in Vector Space: Avenues for the
            Interdisciplinary Study of Meaning Variation},
  year   = {2026},
}
```

---

## License

MIT
