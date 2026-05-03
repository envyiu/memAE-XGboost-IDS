"""Window feature package — re-exports the public API."""
from src.features.window.config import resolve_window_config, window_required_columns
from src.features.window.engine import add_window_features
from src.features.window.names import window_feature_names

__all__ = [
    "add_window_features",
    "window_feature_names",
    "resolve_window_config",
    "window_required_columns",
]
