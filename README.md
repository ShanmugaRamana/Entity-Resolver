# Entity Resolver

Matches business records across three noisy sources. For every record in a deduplicated
reference source (S1), it finds all records in sources S2 and S3 that describe the same business,
at the scale of **2.2M reference records against 10M candidates**, on a single laptop.

The records are messy in every way a real registry is: typos, abbreviations (`Rd`/`Road`,
`Pvt`/`Private`), legal-suffix changes, reordered words and address parts, website domains used
as names, house-number noise, missing addresses, names transliterated from Devanagari and other
scripts, and near-duplicate businesses that are *not* matches. One country appears only in the
data to be matched, never in training.

**Result:** macro F0.5 of **0.986** on 55,000 held-out reference businesses (precision 0.998,
recall 0.964), against a perfect-matcher ceiling of 0.995 for the retrieved candidates.

## Highlights

- **Flipped search direction with a one-owner constraint.** Each S2/S3 record is a query that can
  belong to at most one S1 business, so conflicting merges are impossible by construction.
- **Four retrieval passes** keep 98.5% of true pairs at about 19 candidates per record (reduction
  ratio 0.99998): character trigrams of the name, address words, name + address words, and
  nearest neighbours in a multilingual embedding space.
- **No hand-written language knowledge.** Abbreviations, spelling variants and legal forms are
  mined from training matches; filler words, places and place links (city ↔ region) come from
  each dataset's own statistics. The same code handles a country never seen in training.
- **Evidence from the business's other records.** A record is compared not only with the S1 record
  but with the business's other confidently matched records ("siblings"), and with competing
  businesses of the same name. This separates house-number noise (305 vs 304) from a different
  business (2026 vs 2005).
- **Two-stage LightGBM + cross-encoder.** 77 pair features, 30M training pairs, a second stage
  that sees each query's competing candidates, and a fine-tuned multilingual cross-encoder for
  the uncertain band.
- **A decision layer built for F0.5.** Precision counts double, so the output set per business is
  chosen to maximize expected F0.5, not by a fixed cut-off. Every optional step is tuned together
  with "off" and kept only if it measurably helps.

## How it works

```
 S1 / S2 / S3 TSVs
        │
        ▼
 Normalize ─────── generic: transliteration (anyascii), case, punctuation, domains
        │          learned: token variants, filler words, places, place links
        ▼
 Embed names ───── multilingual-e5-small, fine-tuned on training pairs (original script)
        │
        ▼
 Block ─────────── per country; each S2/S3 record queries the S1 index:
        │          A char-trigram name · B address words · C name+address · D embeddings
        │          union → ≤ 16 per record + top 3 of every pass
        ▼
 Features ──────── 77: TF-IDF / embedding cosines, fuzzy scores, soft token matches,
        │          house-number distance, name rarity, siblings, competitors
        ▼
 Match ─────────── LightGBM stage 1 (3-fold, out-of-fold) → stage 2 (+ query context)
        │          → isotonic calibration → cross-encoder blend on the uncertain band
        ▼
 Decide ────────── each record to its best S1 · per-S1 set with max expected F0.5
        │
        ▼
 matching_results.tsv + candidate_pairs.tsv
```

Details: [docs/methodology.md](docs/methodology.md). How the model got here, step by step:
[docs/development-log.md](docs/development-log.md).

## Results

Validation: 5% of training S1 businesses are held out; half tunes calibration and thresholds,
the other half (55,357 businesses) is only used for reporting. Retrieval runs over the full
training data, so the held-out businesses face the full crowd of distractors.

| Configuration | Macro F0.5 |
|---|---|
| Blocking score only (baseline) | 0.725 |
| Two-stage LightGBM, 58 features | 0.978 |
| + sibling, house-number and competitor features, 2× training data | 0.983 |
| **+ cross-encoder re-scoring (default)** | **0.986** |
| Perfect matcher on the retrieved candidates (ceiling) | 0.995 |

By country: US 0.989, India 0.982. Businesses with no match at all: 0.991. Leave-one-country-out
(train on one country, evaluate on the other): 0.934 and 0.976, the price of an unseen country.

These numbers come from held-out data of the same distribution as training. Data with a different
noise or distractor profile will score lower, and the decision threshold is the first thing to
re-tune there.

## Quick start (synthetic data, about a minute)

```bash
python -m venv venv && source venv/bin/activate
pip install -e .                     # core pipeline (add ".[embed]" for embeddings + cross-encoder)
python scripts/make_synthetic.py --out data_synthetic
python scripts/train.py --data-dir data_synthetic --work-dir work_synthetic --no-embed --jobs 2
python scripts/test.py  --data-dir data_synthetic --work-dir work_synthetic --output-dir output_synthetic
```

On macOS, LightGBM also needs the OpenMP runtime: `brew install libomp`.

## Full run

1. Put the dataset in `dataset/` as described in [dataset/README.md](dataset/README.md).
2. Install everything, including the embedding extras:
   `pip install -e ".[embed]"`, or `pip install -r requirements.txt` for the exact versions used.
3. Train and predict:
   ```bash
   python scripts/train.py      # artifacts + training report in work/model/
   python scripts/test.py       # output/matching_results.tsv and output/candidate_pairs.tsv
   python scripts/score.py --pred work/model/val_matching_results.tsv \
                           --truth work/model/val_ground_truth.tsv      # independent re-check
   ```
   Add `--sample 0.05` to either script for a consistent 5% subset (minutes instead of hours).

Every stage is cached in `work/` with its own key, so an interrupted run resumes, and a settings
change recomputes only the stages it affects.

**Run time** on a MacBook Pro (M4 Pro, 12 cores, 24 GB): training about 3 h, prediction about
2.8 h, peak memory about 20 GB. The first run also downloads the 470 MB encoder.

**Main options** (`scripts/train.py`): `--no-embed` (no torch needed), `--no-finetune`,
`--no-cross-encoder`, `--no-consistency`, `--proxy` (leave-one-country-out report),
`--sim-drop F` (simulate a denser crowd), `--jobs N`, `--force`.

## Repository layout

```
src/er/
  normalize.py     generic text normalization
  vocab.py         learned vocabulary: token variants, filler words, places, place links
  embed.py         encoder fine-tuning, name embeddings, FAISS search
  blocking.py      retrieval passes, union, cap, blocking metrics
  features.py      77 pair features
  model.py         two-stage LightGBM, calibration
  crossenc.py      cross-encoder re-scoring
  decide.py        one-owner assignment + expected-F0.5 set selection
  consistency.py   record-consistency step
  analysis.py      error analysis by slice
  metrics.py       macro F0.5
  pipeline.py      stage runner and per-stage cache
  synthetic.py     synthetic dataset generator
scripts/           train.py, test.py, score.py, make_synthetic.py
tests/             unit tests + end-to-end test on synthetic data
docs/              methodology and development log
dataset/README.md dataset layout (data itself is not in the repo)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not e2e"     # unit tests (seconds)
ruff check src scripts tests
```

CI runs both on every push. Run `pytest -m e2e` by hand after pipeline changes (trains a small
model on synthetic data, about a minute) — it's not part of CI.

## License

MIT, see [LICENSE](LICENSE). The pretrained encoder `intfloat/multilingual-e5-small` is MIT
licensed; all Python dependencies are MIT, BSD, Apache-2.0 or ISC, except `certifi` (MPL-2.0),
which is only used to download the encoder.
