# conrep

Analysis code for:

> Durstewitz, S. J. (2026). *Concept Representations in Vector Space: Avenues for the Interdisciplinary Study of Meaning Variation.*

**conrep** encodes participant-generated word associations as sentence embeddings and models pairwise representational distances to measure how concept meanings vary within and across people.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/conrep
cd conrep
pip install -r requirements.txt
```

Open `companion_analysis.ipynb`, set your data paths and concept lists in **Section 1 (Configuration)**, and run top to bottom.

---

## Repository Structure

```
conrep/
├── companion_analysis.ipynb   ← configure and run here
├── requirements.txt
├── README.md
└── conrep/
    ├── __init__.py            ← public API
    ├── preprocessing.py       ← sentence construction and filtering
    ├── embeddings.py          ← encoding and disk caching
    ├── sharedness.py          ← S_b computation
    ├── mantel.py              ← generic Mantel test
    ├── llm.py                 ← LLM elicitation and comparison
    └── plotting.py            ← all figure functions
```

---

## Data

The notebook expects:

| File | Source |
|---|---|
| `SWOW-EN.complete.20180827.csv` | [smallworldofwords.org](https://smallworldofwords.org/project/research/) — CC BY-NC-ND 3.0 |
| `word_ratings.xlsx` | Brysbaert et al. (2014) concreteness norms |

GeoNames `cities500.txt` is downloaded automatically on first run.

---

## Using Your Own Data

Any dataset that can be structured as `(participantID, cue, R1, R2, R3)` rows works. Pass `noun_filter=False` to `filter_swow()` if your concept set is pre-selected.

To use a different predictor in the Mantel test, pass any `pd.Series` indexed by `participantID` to `build_dyads()`:

```python
my_predictor = df_participants.set_index('participantID')['education_years']
df_dyads = fw.build_dyads(embeddings, predictor=my_predictor, concepts=MY_CONCEPTS)
results  = fw.mantel_test(df_dyads)
```

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
