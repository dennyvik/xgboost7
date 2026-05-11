from __future__ import annotations

from typing import Any


FEATURE_REGISTRY: dict[str, dict[str, str]] = {
    "atr_14": {"group": "volatility", "version": "1.0"},
    "atr_28": {"group": "volatility", "version": "1.0"},
    "ret_3": {"group": "momentum", "version": "1.0"},
    "ret_6": {"group": "momentum", "version": "1.0"},
    "range_ratio": {"group": "structure", "version": "1.0"},
    "ema_5_close": {"group": "trend", "version": "1.0"},
    "ema_20_close": {"group": "trend", "version": "1.0"},
    "close_to_ema_5_ratio": {"group": "trend", "version": "1.0"},
    "close_to_ema_20_ratio": {"group": "trend", "version": "1.0"},
    "macd_12_26_9": {"group": "momentum", "version": "1.0"},
    "macd_hist_12_26_9": {"group": "momentum", "version": "1.0"},
    "macd_signal_12_26_9": {"group": "momentum", "version": "1.0"},
    "macd_8_21_5": {"group": "momentum", "version": "1.0"},
    "macd_hist_8_21_5": {"group": "momentum", "version": "1.0"},
    "macd_signal_8_21_5": {"group": "momentum", "version": "1.0"},
    "rsi_14": {"group": "momentum", "version": "1.0"},
    "rsi_7": {"group": "momentum", "version": "1.0"},
    "stoch_k_14_3_3": {"group": "momentum", "version": "1.0"},
    "stoch_d_14_3_3": {"group": "momentum", "version": "1.0"},
    "stoch_k_9_3_3": {"group": "momentum", "version": "1.0"},
    "stoch_d_9_3_3": {"group": "momentum", "version": "1.0"},
    "event_vol_spike": {"group": "event", "version": "1.0"},
    "event_impulse": {"group": "event", "version": "1.0"},
}


def get_active_features(config: dict[str, Any]) -> list[str]:
    """Return the ordered list of registry feature names enabled by config.

    Resolution order:
    1. ``config["features"]["enabled"]`` — explicit allow-list of feature names.
    2. ``config["features"]["groups"]`` — allow-list of feature groups.
    3. No ``features`` section in config — all registry features are active.
    """
    features_config = config.get("features", {})

    enabled = features_config.get("enabled")
    if enabled is not None:
        return [name for name in enabled if name in FEATURE_REGISTRY]

    groups = features_config.get("groups")
    if groups is not None:
        return [
            name
            for name, meta in FEATURE_REGISTRY.items()
            if meta["group"] in groups
        ]

    return list(FEATURE_REGISTRY.keys())


def describe_active_features(config: dict[str, Any]) -> list[dict[str, str]]:
    """Return metadata for each active feature (name, group, version)."""
    return [
        {"name": name, **FEATURE_REGISTRY[name]}
        for name in get_active_features(config)
    ]
