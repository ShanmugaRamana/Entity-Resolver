# Data

The dataset is not part of this repository. Download `dataset.zip` (about 1 GB) from the
[dataset download page](https://shanmugaramana.github.io/Entity-Resolver/dataset.html) and unzip
it in the repository root: it unpacks into this folder. You can also put the data anywhere and
pass `--data-dir` / set `ER_DATA_DIR`. Either way it needs this layout:

```
dataset/
├── train/
│   ├── train_source1.tsv         reference records (deduplicated)
│   ├── train_source2.tsv         records from source 2
│   ├── train_source3.tsv         records from source 3
│   └── train_ground_truth.tsv    which source-2/3 records belong to each source-1 record
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are **tab-separated** with a header row (names and addresses contain commas).

**Source files**

| Column | Content |
|---|---|
| `entity_id` | unique id; the prefix gives the source: `S1-`, `S2-`, `S3-` |
| `business_name` | noisy business name (typos, abbreviations, legal-suffix changes, other scripts) |
| `business_address` | noisy address (reordered or missing parts, abbreviations, house-number noise); may be empty |
| `country` | country label, treated as an open set (a country may appear only in `test/`) |

**Ground truth** (`train_ground_truth.tsv`)

| Column | Content |
|---|---|
| `source1_entity_id` | a source-1 id |
| `matched_entity_ids` | comma-separated source-2/3 ids of the same business; empty if none |

Each source-2/3 record belongs to at most one source-1 record.

**Outputs** written by `scripts/test.py` use the same two-column format:
`matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) and `candidate_pairs.tsv`
(`source1_entity_id`, `candidate_entity_ids`: the candidates the model scored).

## No data? Use the synthetic generator

```bash
python scripts/make_synthetic.py --out data_synthetic
```

writes a small dataset in exactly this layout, with the same kinds of noise, so the full pipeline
can be tried in about a minute (see the main README).
