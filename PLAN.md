# Milestone 1 — Implementation Plan

## Naming Conventions + Logging + Project Wiring

---

# 1. Objective

Establish a **consistent, debuggable ML pipeline foundation** with:

* strict naming conventions
* structured logging
* reproducible experiment runs
* modular code layout

This plan must be implemented **before optimizing models**.

---

# 2. Project Structure (Create First)

```text
trading-ml/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── data/
│   ├── features/
│   ├── events/
│   ├── labeling/
│   ├── dataset/
│   ├── models/
│   ├── evaluation/
│   ├── diagnostics/
│   ├── utils/
│   └── pipelines/
│
├── outputs/
│   ├── runs/
│   └── debug/
│
├── scripts/
│   └── run_train.py
│
├── tests/
├── requirements.txt
└── README.md
```

---

# 3. Naming Conventions (Enforce Globally)

## 3.1 General Rules

* Use `snake_case` everywhere
* No vague names: `temp`, `final`, `new`, `df2` are forbidden
* Names must encode meaning clearly

---

## 3.2 DataFrame Naming

```python
df_raw
df_feat
df_event
df_labeled
df_train
df_val
df_test
```

---

## 3.3 Column Naming

### Labels

```python
target_opportunity
```

---

### Predictions

```python
p_opportunity
rank_opportunity
```

---

### Event Signals (mandatory prefix)

```python
event_vol_spike
event_compression
event_breakout_attempt
event_impulse
```

---

## 3.4 Function Naming Pattern

```text
verb_object_context
```

Examples:

```python
load_csv_data()
validate_ohlc_data()

compute_atr()
compute_ema_features()

detect_volatility_spike()
detect_compression_event()

create_opportunity_label()

train_xgb_model()
predict_opportunity()

calculate_precision_at_k()
compute_event_statistics()
```

---

## 3.5 File Naming

```text
<domain>_<purpose>.py
```

Examples:

```text
volatility_features.py
trend_features.py
event_detection.py
opportunity_labeling.py
model_training.py
evaluation_metrics.py
event_diagnostics.py
```

---

# 4. Run Management (Reproducibility)

## 4.1 Run ID Format

```text
<model>_<dataset>_<YYYY-MM-DD>_<HHMM>
```

Example:

```text
xgb6_m1_2026-05-06_1432
```

---

## 4.2 Run Directory Structure

```text
outputs/runs/<run_id>/
    config.yaml
    logs.txt
    metrics.json
    model.pkl
    feature_importance.csv
```

---

## 4.3 Requirement

* Every run MUST save its config copy
* No run without a unique run_id
* No overwriting previous runs

---

# 5. Logging System (Mandatory)

## 5.1 Log Levels

```text
INFO
DEBUG
WARNING
ERROR
```

---

## 5.2 Log Format

```text
timestamp | level | module | message | key=value pairs
```

Example:

```text
2026-05-06 14:32:10 | INFO | data.loader | loaded data | rows=1171415
```

---

## 5.3 Logger Setup (`src/utils/logger.py`)

```python
import logging
import os

def setup_logger(run_dir: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    fh = logging.FileHandler(os.path.join(run_dir, "logs.txt"))
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
```

---

## 5.4 Structured Logging Helper

```python
def log_kv(logger, message, **kwargs):
    kv = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"{message} | {kv}")
```

---

# 6. Required Logging Coverage

## 6.1 Data Stage

Log:

```text
rows
date_range
missing_values
```

---

## 6.2 Feature Stage

Log:

```text
feature_count
nan_count
```

---

## 6.3 Label Stage

Log:

```text
label_distribution
positive_rate
```

---

## 6.4 Training Stage

Log:

```text
train_size
val_size
best_iteration
validation_score
```

---

## 6.5 Evaluation Stage

Log:

```text
precision_at_k
baseline
candidate_rate
```

---

## 6.6 Event Diagnostics

Log per event:

```text
event_name
frequency
trade_rate
precision_at_k
```

---

# 7. Debugging Infrastructure

## 7.1 Intermediate Snapshots

Save:

```text
outputs/debug/
    df_feat_sample.csv
    df_labeled_sample.csv
```

---

## 7.2 Assertions (Mandatory)

```python
assert df.shape[0] > 0
assert "target_opportunity" in df.columns
```

---

## 7.3 Isolation Rule

Each stage must be runnable independently:

```text
data → features → labels → model → evaluation
```

---

# 8. Pipeline Entry Point

## `scripts/run_train.py`

```python
from src.pipelines.train_pipeline import run_pipeline
from src.utils.config import load_config

config = load_config("configs/config.yaml")
run_pipeline(config)
```

---

## `src/pipelines/train_pipeline.py`

Must orchestrate:

```text
1. load data
2. build features
3. build event signals
4. create labels
5. split dataset
6. downsample training data
7. train model
8. predict
9. evaluate
10. run event diagnostics
11. save outputs
```

---

# 9. Anti-Patterns (Strictly Forbidden)

* Using `print()` instead of logger
* Silent exception handling (`except: pass`)
* Reusing variable names across stages
* Mixing multiple pipeline steps in one function
* Overwriting outputs without run_id

---

# 10. Definition of Done

This setup is complete when:

* One full pipeline run executes successfully
* Logs clearly show each stage
* Outputs are saved under unique run_id
* Intermediate debug files are inspectable
* Metrics + event diagnostics are reproducible

---

# Final Instruction

Do NOT optimize model performance yet.

First goal:

```text
make the system observable, traceable, and stable
```

Without this, any “improvement” is unreliable.
