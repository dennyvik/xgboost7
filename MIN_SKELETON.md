# Minimal Working Implementation — Skeleton

## 0. Install

```bash
pip install pandas numpy scikit-learn xgboost pyyaml
```

---

# 1. `configs/config.yaml`

```yaml
data:
  path: data/raw/xauusd_m1.csv

split:
  train_end: 2025-04-01
  val_end: 2025-10-01

sampling:
  ratio: 8

model:
  max_depth: 4
  learning_rate: 0.05
  n_estimators: 200
  scale_pos_weight: 8

evaluation:
  top_k: [0.5, 1, 2]
```

---

# 2. `src/utils/config.py`

```python
import yaml

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
```

---

# 3. `src/utils/logger.py`

```python
import logging, os

def setup_logger(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(os.path.join(run_dir, "logs.txt"))
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def log_kv(logger, message, **kwargs):
    kv = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"{message} | {kv}")
```

---

# 4. `src/data/loader.py`

```python
import pandas as pd

def load_csv_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df
```

---

# 5. `src/features/builder.py`

```python
import pandas as pd
import numpy as np

def compute_atr(df, window=14):
    high_low = df["high"] - df["low"]
    return high_low.rolling(window).mean()

def compute_returns(df, window):
    return df["close"].pct_change(window)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["atr_14"] = compute_atr(df, 14)
    df["atr_28"] = compute_atr(df, 28)

    df["ret_3"] = compute_returns(df, 3)
    df["ret_6"] = compute_returns(df, 6)

    df["range_ratio"] = (df["high"] - df["low"]) / df["close"]

    return df
```

---

# 6. `src/events/builder.py`

```python
def build_event_signals(df):
    df = df.copy()

    # Simple baseline rules (causal)
    df["event_vol_spike"] = (df["atr_14"] > df["atr_14"].rolling(50).mean()).astype(int)

    df["event_impulse"] = (df["ret_3"].abs() > df["ret_3"].rolling(50).std()).astype(int)

    return df
```

---

# 7. `src/labeling/opportunity.py`

```python
import numpy as np

def create_opportunity_label(df):
    df = df.copy()

    # Simple proxy: large move next N bars (placeholder)
    future_return = df["close"].shift(-5) / df["close"] - 1

    df["target_opportunity"] = (future_return.abs() > 0.002).astype(int)

    return df
```

---

# 8. `src/dataset/builder.py`

```python
def time_split(df, config):
    train_end = config["split"]["train_end"]
    val_end = config["split"]["val_end"]

    df_train = df[df["datetime"] <= train_end]
    df_val = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)]
    df_test = df[df["datetime"] > val_end]

    return df_train, df_val, df_test
```

---

# 9. `src/dataset/sampler.py`

```python
import pandas as pd

def downsample(df, ratio):
    pos = df[df["target_opportunity"] == 1]
    neg = df[df["target_opportunity"] == 0]

    neg_sample = neg.sample(len(pos) * ratio, random_state=42)

    return pd.concat([pos, neg_sample]).sample(frac=1, random_state=42)
```

---

# 10. `src/models/trainer.py`

```python
from xgboost import XGBClassifier

def train_model(X_train, y_train, X_val, y_val, config):
    model = XGBClassifier(
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        n_estimators=config["model"]["n_estimators"],
        scale_pos_weight=config["model"]["scale_pos_weight"],
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model

def predict(model, X):
    return model.predict_proba(X)[:, 1]
```

---

# 11. `src/evaluation/metrics.py`

```python
import numpy as np

def precision_at_k(df, k_percent):
    n = int(len(df) * (k_percent / 100))
    top = df.nlargest(n, "p_opportunity")
    return top["target_opportunity"].mean()
```

---

# 12. `src/diagnostics/event_analysis.py`

```python
def compute_event_stats(df):
    results = []

    for col in [c for c in df.columns if c.startswith("event_")]:
        subset = df[df[col] == 1]

        if len(subset) == 0:
            continue

        results.append({
            "event": col,
            "frequency": len(subset),
            "trade_rate": subset["target_opportunity"].mean()
        })

    return results
```

---

# 13. `src/pipelines/train_pipeline.py`

```python
import os, datetime
from src.data.loader import load_csv_data
from src.features.builder import build_features
from src.events.builder import build_event_signals
from src.labeling.opportunity import create_opportunity_label
from src.dataset.builder import time_split
from src.dataset.sampler import downsample
from src.models.trainer import train_model, predict
from src.evaluation.metrics import precision_at_k
from src.diagnostics.event_analysis import compute_event_stats
from src.utils.logger import setup_logger, log_kv

def run_pipeline(config):

    run_id = datetime.datetime.now().strftime("xgb6_m1_%Y-%m-%d_%H%M")
    run_dir = f"outputs/runs/{run_id}"

    logger = setup_logger(run_dir)

    # Load
    df = load_csv_data(config["data"]["path"])
    log_kv(logger, "data_loaded", rows=len(df))

    # Features
    df = build_features(df)
    log_kv(logger, "features_built", cols=len(df.columns))

    # Events
    df = build_event_signals(df)

    # Label
    df = create_opportunity_label(df)

    # Drop NA
    df = df.dropna()
    log_kv(logger, "after_dropna", rows=len(df))

    # Split
    train, val, test = time_split(df, config)

    # Downsample
    train = downsample(train, config["sampling"]["ratio"])

    # Prepare X/y
    features = [c for c in df.columns if c not in ["datetime", "target_opportunity"]]

    X_train, y_train = train[features], train["target_opportunity"]
    X_val, y_val = val[features], val["target_opportunity"]
    X_test, y_test = test[features], test["target_opportunity"]

    # Train
    model = train_model(X_train, y_train, X_val, y_val, config)

    # Predict
    val["p_opportunity"] = predict(model, X_val)
    test["p_opportunity"] = predict(model, X_test)

    # Evaluate
    for k in config["evaluation"]["top_k"]:
        p = precision_at_k(val, k)
        log_kv(logger, "precision_at_k", k=k, value=p)

    # Event diagnostics
    stats = compute_event_stats(val)
    for s in stats:
        log_kv(logger, "event_stat", **s)
```

---

# 14. `scripts/run_train.py`

```python
from src.utils.config import load_config
from src.pipelines.train_pipeline import run_pipeline

config = load_config("configs/config.yaml")
run_pipeline(config)
```

---

# 15. What You Should See

After running:

```bash
python scripts/run_train.py
```

You should get:

```text id="nyvqg4"
outputs/runs/xgb6_m1_YYYY-MM-DD_HHMM/
    logs.txt
```

Logs like:

```text id="ytzwak"
INFO data_loaded | rows=1171415
INFO features_built | cols=...
INFO precision_at_k | k=1 value=...
INFO event_stat | event=event_vol_spike ...
```

---

# Final Note

This is intentionally **minimal**:

* labels are simple → replace later
* events are basic → refine later
* evaluation is limited → expand later

But the **structure is correct**.
That’s what matters right now.

---

If you want next step, I can:

* upgrade this into **production-grade (with artifacts saving, metrics export, feature importance)**
* or help you **debug your first real run output against expectations**
