"""Backward-compatibility shim — all logic moved to src.features.window package."""
from src.features.window import (  # noqa: F401
    add_window_features,
    resolve_window_config,
    window_feature_names,
    window_required_columns,
)
