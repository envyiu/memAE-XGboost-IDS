"""Window feature configuration: defaults, merging, and column resolution."""
from __future__ import annotations

from typing import Any


DEFAULT_WINDOW_CONFIG: dict[str, Any] = {
    "enabled": True,
    "group_by": ["source_file", "source_ip"],
    "order_by": ["timestamp", "row_id"],
    "source_ip_column": "source_ip",
    "source_port_column": "source_port",
    "destination_ip_column": "destination_ip",
    "timestamp_column": "timestamp",
    "destination_port_column": "destination_port",
    "fallback_context_key_column": "protocol",
    "window_sizes": [10, 50, 200, 1000],
    "time_window_seconds": [60, 300, 600, 3600],
    "include_flow_count": True,
    "include_unique_destination_port": True,
    "include_destination_port_count": True,
    "include_unique_destination_ip": True,
    "include_destination_ip_count": True,
    "include_destination_service_count": True,
    "include_flag_ratios": True,
    "include_packet_byte_sums": True,
    "include_unique_port_ratio": True,
    "include_service_context": True,
    "include_behavior_proxies": True,
    "include_periodicity": False,
    "include_botnet_context": False,
    "include_low_slow": False,
    "include_source_port_context": False,
    "include_watched_ports": False,
    "include_port_diversity": False,
    "include_timing_regularity": False,
    "include_dest_concentration": False,
    "short_flow_packet_threshold": 6,
    "small_flow_byte_threshold": 512,
    "burst_gap_seconds": 1.0,
    "watched_ports": [8080],
    "service_family_ports": {
        "auth": [21, 22, 23, 25, 110, 143, 3389],
        "web": [80, 443, 8080, 8443],
        "infra": [53, 67, 68, 123, 161],
        "fileshare": [135, 137, 138, 139, 445],
        "database": [1433, 1521, 3306, 5432, 6379],
    },
    "flag_columns": {
        "syn": "syn_flag_count",
        "ack": "ack_flag_count",
        "rst": "rst_flag_count",
        "fin": "fin_flag_count",
    },
    "packet_columns": {
        "fwd": "total_fwd_packets",
        "bwd": "total_backward_packets",
    },
    "byte_columns": {
        "fwd": "total_length_of_fwd_packets",
        "bwd": "total_length_of_bwd_packets",
    },
}


def merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user config over defaults, handling nested dicts."""
    merged = dict(DEFAULT_WINDOW_CONFIG)
    if not config:
        return merged
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def resolve_window_config(config: dict[str, Any] | None, available_columns: set[str]) -> dict[str, Any]:
    """Resolve the effective context key column, falling back if needed."""
    cfg = merge_config(config)
    context_col = cfg["destination_port_column"]
    if context_col in available_columns:
        cfg["effective_context_key_column"] = context_col
        cfg["effective_context_key_source"] = "destination_port"
        return cfg

    fallback = cfg.get("fallback_context_key_column")
    if fallback and fallback in available_columns:
        cfg["destination_port_column"] = fallback
        cfg["effective_context_key_column"] = fallback
        cfg["effective_context_key_source"] = f"fallback:{fallback}"
        cfg["include_service_context"] = False
        return cfg

    cfg["effective_context_key_column"] = None
    cfg["effective_context_key_source"] = "missing"
    cfg["include_service_context"] = False
    cfg["include_unique_destination_port"] = False
    cfg["include_destination_port_count"] = False
    cfg["include_unique_port_ratio"] = False
    return cfg


def window_required_columns(config: dict[str, Any]) -> list[str]:
    """Return sorted list of DataFrame columns required for the given config."""
    cfg = merge_config(config)
    cols = set(cfg["group_by"]) | set(cfg["order_by"])
    if (cfg["include_unique_destination_port"] or cfg["include_destination_port_count"]) and cfg["destination_port_column"]:
        cols.add(cfg["destination_port_column"])
    if cfg.get("include_source_port_context") or cfg.get("include_botnet_context") or cfg.get("include_watched_ports"):
        cols.add(cfg["source_port_column"])
    if cfg.get("include_unique_destination_ip") or cfg.get("include_destination_ip_count") or cfg.get("include_destination_service_count") or cfg.get("include_dest_concentration"):
        cols.add(cfg["destination_ip_column"])
    if cfg.get("include_port_diversity") or cfg.get("include_dest_concentration"):
        cols.add(cfg["destination_port_column"])
    if cfg["include_packet_byte_sums"]:
        cols.update(cfg["packet_columns"].values())
        cols.update(cfg["byte_columns"].values())
    if cfg["include_flag_ratios"]:
        cols.update(cfg["flag_columns"].values())
    return sorted(cols)
