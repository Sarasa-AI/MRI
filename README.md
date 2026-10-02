# RSNA Knee Abnormality Detection 2026 — Control Plane

This repository is the **Mac control plane** for the RSNA Knee 2026
competition entry. It holds source, configs, tests, experiment definitions,
and documentation.

Competition DICOMs, training, inference, and submission generation run on
**Kaggle** (data/compute plane). Do **not** download or copy the RSNA dataset
onto this Mac.

## Quick start (local, metadata only)

```bash
cd /Users/sarasa/Documents/MRI/MRI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_smoke_local.py
python scripts/validate_submission.py data/fixtures/sample_submission.csv \
  --expected-ids-csv data/fixtures/test.csv
```

## Layout

```
src/rsna_knee/     package (data, datasets, models, training, inference,
                   evaluation, labels, utils, runtime, submission)
configs/           YAML configs (runtime local|kaggle, experiments)
experiments/       registry, cards, results
notebooks/kaggle/  Kaggle notebook templates
scripts/           local smoke + submission validator CLIs
tests/             unit tests (fixtures only)
docs/              architecture, gates, experiment protocol
data/fixtures/     synthetic CSVs for local tests (not competition data)
```

## Canonical docs

- [PROJECT_STATE.md](PROJECT_STATE.md) — verified status
- [ROADMAP.md](ROADMAP.md) — near-term plan
- [DECISIONS.md](DECISIONS.md) — durable decisions
- [EXPERIMENTS.md](EXPERIMENTS.md) — experiment queue
- [RSNA-G1-KAGGLE-DATA-ACCESS.md](RSNA-G1-KAGGLE-DATA-ACCESS.md) — G1 probe
- [docs/architecture.md](docs/architecture.md) — Mac vs Kaggle split
- [docs/gates.md](docs/gates.md) — G1→G8 gate board

## Historical note

Prior to 2026-10-02 this README described the repo as a portfolio-only control
plane with RSNA implementation expected in `SARASA_kaggle`. That path is a
heart-disease tabular project, not RSNA knee. Decision **D-006** co-locates the
RSNA engineering skeleton here under the Mac-control / Kaggle-compute
architecture established in G1.
