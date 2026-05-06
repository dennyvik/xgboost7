# XGBoost7 Trading ML Foundation

This project is the Milestone 1 foundation for an observable, reproducible XGBoost pipeline on XAU/USD M1 data. The goal is structure, logging, artifacts, and repeatability before model-performance work.

## Setup

From the project root:

```bash
python3.11 -m venv venv311
source venv311/bin/activate
python -m pip install -r requirements.txt
```

## Data Contract

The default input file is:

```text
data/raw/XAUUSD_M1.csv
```

The loader accepts either canonical columns:

```text
datetime, open, high, low, close
```

or MetaTrader-style whitespace-separated columns:

```text
<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> ...
```

All data is sorted by `datetime` before feature generation.

## Run Training

```bash
source venv311/bin/activate
python scripts/run_train.py
```

Each run creates a unique directory under `outputs/runs/` using configurable naming from `configs/config.yaml`:

```text
outputs/runs/<model>_<dataset>_<YYYY-MM-DD>_<HHMM>/
    config.yaml
    logs.txt
    metrics.json
    model.pkl
    feature_importance.csv
```

Debug snapshots are written to:

```text
outputs/debug/df_feat_sample.csv
outputs/debug/df_labeled_sample.csv
```

## Tests

```bash
source venv311/bin/activate
python -m pytest
```

## Milestone Rule

Do not optimize model performance yet. First make the system observable, traceable, and stable.