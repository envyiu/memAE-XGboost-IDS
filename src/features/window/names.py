"""Feature name generation from window configuration."""
from __future__ import annotations

from typing import Any

from src.features.window.config import merge_config


def window_feature_names(config: dict[str, Any]) -> list[str]:
    """Generate the full ordered list of feature column names for a given config."""
    cfg = merge_config(config)
    sizes = [int(s) for s in cfg["window_sizes"]]
    time_sizes = [int(s) for s in cfg.get("time_window_seconds", [])]
    names: list[str] = []

    # --- Static context features ---
    if cfg.get("include_service_context", True):
        names.extend([
            "ctx_dest_port_is_system",
            "ctx_dest_port_is_registered",
            "ctx_dest_port_is_dynamic",
        ])
        for family in sorted(cfg.get("service_family_ports", {})):
            names.append(f"ctx_service_{family}")
    if cfg.get("include_source_port_context"):
        names.extend([
            "ctx_source_port_is_system",
            "ctx_source_port_is_registered",
            "ctx_source_port_is_dynamic",
            "ctx_source_dest_same_port",
        ])
        for family in sorted(cfg.get("service_family_ports", {})):
            names.append(f"ctx_source_service_{family}")
    if cfg.get("include_watched_ports"):
        for port in sorted(int(p) for p in cfg.get("watched_ports", [])):
            names.append(f"ctx_source_port_{port}")
            names.append(f"ctx_dest_port_{port}")

    # --- Count-based window features ---
    for window in sizes:
        if cfg["include_flow_count"]:
            names.append(f"win_flow_count_{window}")
        if cfg["include_unique_destination_port"]:
            names.append(f"win_unique_destination_port_{window}")
        if cfg["include_destination_port_count"]:
            names.append(f"win_destination_port_count_{window}")
        if cfg["include_unique_port_ratio"] and cfg["include_unique_destination_port"]:
            names.append(f"win_unique_destination_port_ratio_{window}")
            names.append(f"win_same_destination_port_ratio_{window}")
            names.append(f"win_new_destination_port_indicator_{window}")
        if cfg["include_packet_byte_sums"]:
            names.append(f"win_packets_sum_{window}")
            names.append(f"win_bytes_sum_{window}")
        if cfg["include_flag_ratios"]:
            names.append(f"win_syn_ratio_{window}")
            names.append(f"win_rst_ratio_{window}")
            names.append(f"win_fin_ratio_{window}")
        if cfg.get("include_behavior_proxies", True):
            names.append(f"win_port_switch_ratio_{window}")
            names.append(f"win_short_flow_ratio_{window}")
            names.append(f"win_small_bytes_ratio_{window}")
            names.append(f"win_syn_dominant_ratio_{window}")
            names.append(f"win_rst_dominant_ratio_{window}")
        if cfg.get("include_periodicity"):
            names.append(f"win_burst_count_{window}")
        if cfg.get("include_botnet_context"):
            names.append(f"win_dest_repeat_ratio_{window}")
            names.append(f"win_source_port_repeat_ratio_{window}")
            names.append(f"win_source_dest_port_pair_repeat_ratio_{window}")
        if cfg.get("include_low_slow"):
            names.append(f"win_low_slow_repeat_count_{window}")
        if cfg.get("include_port_diversity"):
            names.append(f"win_port_entropy_{window}")
            names.append(f"win_port_concentration_{window}")
            names.append(f"win_sequential_port_ratio_{window}")
            names.append(f"win_high_port_ratio_{window}")
        if cfg.get("include_timing_regularity"):
            names.append(f"win_inter_arrival_cv_{window}")
            names.append(f"win_inter_arrival_regularity_{window}")
        if cfg.get("include_dest_concentration"):
            names.append(f"win_dest_ip_entropy_{window}")
            names.append(f"win_single_dest_ratio_{window}")
            names.append(f"win_port_per_dest_ip_{window}")

    # --- Time-based window features ---
    for seconds in time_sizes:
        names.append(f"time_flow_count_{seconds}s")
        if cfg["include_unique_destination_port"]:
            names.append(f"time_unique_destination_port_{seconds}s")
        if cfg["include_destination_port_count"]:
            names.append(f"time_destination_port_count_{seconds}s")
        if cfg["include_unique_destination_ip"]:
            names.append(f"time_unique_destination_ip_{seconds}s")
        if cfg["include_destination_ip_count"]:
            names.append(f"time_destination_ip_count_{seconds}s")
        if cfg["include_destination_service_count"]:
            names.append(f"time_destination_service_count_{seconds}s")
        if cfg["include_unique_port_ratio"] and cfg["include_unique_destination_port"]:
            names.append(f"time_unique_destination_port_ratio_{seconds}s")
        if cfg["include_packet_byte_sums"]:
            names.append(f"time_packets_sum_{seconds}s")
            names.append(f"time_bytes_sum_{seconds}s")
        if cfg["include_flag_ratios"]:
            names.append(f"time_syn_ratio_{seconds}s")
            names.append(f"time_rst_ratio_{seconds}s")
            names.append(f"time_fin_ratio_{seconds}s")
        if cfg.get("include_behavior_proxies", True):
            names.append(f"time_port_switch_ratio_{seconds}s")
            names.append(f"time_short_flow_ratio_{seconds}s")
            names.append(f"time_small_bytes_ratio_{seconds}s")
            names.append(f"time_syn_dominant_ratio_{seconds}s")
            names.append(f"time_rst_dominant_ratio_{seconds}s")
        if cfg.get("include_periodicity"):
            names.append(f"time_burst_count_{seconds}s")
        if cfg.get("include_botnet_context"):
            names.append(f"time_dest_repeat_ratio_{seconds}s")
            names.append(f"time_source_port_repeat_ratio_{seconds}s")
            names.append(f"time_source_dest_port_pair_repeat_ratio_{seconds}s")
        if cfg.get("include_low_slow"):
            names.append(f"time_low_slow_repeat_count_{seconds}s")
        if cfg.get("include_port_diversity"):
            names.append(f"time_port_entropy_{seconds}s")
            names.append(f"time_port_concentration_{seconds}s")
            names.append(f"time_sequential_port_ratio_{seconds}s")
            names.append(f"time_high_port_ratio_{seconds}s")
        if cfg.get("include_timing_regularity"):
            names.append(f"time_inter_arrival_cv_{seconds}s")
            names.append(f"time_inter_arrival_regularity_{seconds}s")
        if cfg.get("include_dest_concentration"):
            names.append(f"time_dest_ip_entropy_{seconds}s")
            names.append(f"time_single_dest_ratio_{seconds}s")
            names.append(f"time_port_per_dest_ip_{seconds}s")
    return names
