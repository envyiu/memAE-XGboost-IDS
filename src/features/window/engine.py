"""Main orchestration: add_window_features and count-based window logic."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from src.features.window.config import merge_config, window_required_columns
from src.features.window.names import window_feature_names
from src.features.window.rolling import (
    MAX_Q,
    entropy,
    get_f_log_f,
    rolling_binary_ratio,
    rolling_sum,
    to_unix_seconds,
    window_port_stats,
    window_value_counts,
)
from src.features.window.time_window import time_window_stats


def add_window_features(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Compute all window features and append them to the DataFrame."""
    cfg = merge_config(config)
    if not cfg.get("enabled", True):
        return df, []

    _F_LOG_F_CACHE = get_f_log_f()

    required = window_required_columns(cfg)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for window features: {missing}")

    group_by = list(cfg["group_by"])
    order_by = list(cfg["order_by"])
    timestamp_col = cfg.get("timestamp_column")
    source_port_col = cfg.get("source_port_column")
    dest_ip_col = cfg.get("destination_ip_column")
    dest_port_col = cfg["destination_port_column"]
    window_sizes = [int(s) for s in cfg["window_sizes"]]
    time_window_seconds = [int(s) for s in cfg.get("time_window_seconds", [])]
    burst_gap_seconds = float(cfg.get("burst_gap_seconds", 1.0))
    eps = 1e-6

    original_row_ids = df["row_id"].to_numpy(copy=True) if "row_id" in df.columns else None
    df_sorted = df.reset_index(drop=True).copy()
    temp_timestamp_col = None
    if timestamp_col and timestamp_col in df_sorted.columns:
        temp_timestamp_col = "__timestamp_for_sort"
        df_sorted[temp_timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], errors="coerce", dayfirst=True)
        if df_sorted[temp_timestamp_col].isna().all():
            df_sorted[temp_timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], errors="coerce", dayfirst=False)
    sort_cols = group_by + [temp_timestamp_col if col == timestamp_col and temp_timestamp_col else col for col in order_by]
    df_sorted = df_sorted.sort_values(sort_cols).reset_index(drop=True)

    n_rows = len(df_sorted)
    new_columns = window_feature_names(cfg)
    data: dict[str, np.ndarray] = {name: np.zeros(n_rows, dtype=np.float32) for name in new_columns}

    packet_total = None
    byte_total = None
    if cfg["include_packet_byte_sums"]:
        packet_fwd = cfg["packet_columns"]["fwd"]
        packet_bwd = cfg["packet_columns"]["bwd"]
        byte_fwd = cfg["byte_columns"]["fwd"]
        byte_bwd = cfg["byte_columns"]["bwd"]
        packet_total = (
            df_sorted[packet_fwd].to_numpy(dtype=np.float32)
            + df_sorted[packet_bwd].to_numpy(dtype=np.float32)
        )
        byte_total = (
            df_sorted[byte_fwd].to_numpy(dtype=np.float32)
            + df_sorted[byte_bwd].to_numpy(dtype=np.float32)
        )
        packet_total = np.nan_to_num(packet_total, nan=0.0)
        byte_total = np.nan_to_num(byte_total, nan=0.0)

    syn_arr = rst_arr = fin_arr = ack_arr = None
    if cfg["include_flag_ratios"]:
        flags = cfg["flag_columns"]
        syn_arr = np.nan_to_num(df_sorted[flags["syn"]].to_numpy(dtype=np.float32), nan=0.0)
        rst_arr = np.nan_to_num(df_sorted[flags["rst"]].to_numpy(dtype=np.float32), nan=0.0)
        fin_arr = np.nan_to_num(df_sorted[flags["fin"]].to_numpy(dtype=np.float32), nan=0.0)
        ack_arr = np.nan_to_num(df_sorted[flags["ack"]].to_numpy(dtype=np.float32), nan=0.0)

    if cfg.get("include_service_context", True):
        ports_all = df_sorted[dest_port_col].fillna(-1).to_numpy(dtype=np.int32)
        data["ctx_dest_port_is_system"] = (ports_all < 1024).astype("float32")
        data["ctx_dest_port_is_registered"] = ((ports_all >= 1024) & (ports_all < 49152)).astype("float32")
        data["ctx_dest_port_is_dynamic"] = (ports_all >= 49152).astype("float32")
        for family, ports in sorted(cfg.get("service_family_ports", {}).items()):
            data[f"ctx_service_{family}"] = np.isin(ports_all, np.asarray(ports, dtype=np.int32)).astype("float32")

    source_ports_all = None
    if source_port_col and source_port_col in df_sorted.columns:
        source_ports_all = df_sorted[source_port_col].fillna(-1).to_numpy(dtype=np.int32)
    if cfg.get("include_source_port_context") and source_ports_all is not None:
        data["ctx_source_port_is_system"] = (source_ports_all < 1024).astype("float32")
        data["ctx_source_port_is_registered"] = ((source_ports_all >= 1024) & (source_ports_all < 49152)).astype("float32")
        data["ctx_source_port_is_dynamic"] = (source_ports_all >= 49152).astype("float32")
        data["ctx_source_dest_same_port"] = (source_ports_all == df_sorted[dest_port_col].fillna(-1).to_numpy(dtype=np.int32)).astype("float32")
        for family, ports in sorted(cfg.get("service_family_ports", {}).items()):
            data[f"ctx_source_service_{family}"] = np.isin(source_ports_all, np.asarray(ports, dtype=np.int32)).astype("float32")
    if cfg.get("include_watched_ports") and source_ports_all is not None:
        dest_ports_all = df_sorted[dest_port_col].fillna(-1).to_numpy(dtype=np.int32)
        for port in sorted(int(p) for p in cfg.get("watched_ports", [])):
            data[f"ctx_source_port_{port}"] = (source_ports_all == port).astype("float32")
            data[f"ctx_dest_port_{port}"] = (dest_ports_all == port).astype("float32")

    grouped_indices = df_sorted.groupby(group_by, sort=False).indices
    for idxs in grouped_indices.values():
        idxs = np.asarray(idxs, dtype=np.int64)
        group_len = len(idxs)
        if group_len == 0:
            continue

        if cfg["include_flow_count"]:
            positions = np.arange(group_len, dtype=np.int32) + 1
            for window in window_sizes:
                data[f"win_flow_count_{window}"][idxs] = np.minimum(positions, window).astype("float32")

        port_counts = unique_counts = None
        dest_counts_by_window = None
        ports = None
        source_ports = None
        dest_ips = None
        if source_port_col and source_port_col in df_sorted.columns:
            source_ports = df_sorted.loc[idxs, source_port_col].fillna(-1).to_numpy(dtype=np.int32)
        if cfg["include_unique_destination_port"] or cfg["include_destination_port_count"] or cfg.get("include_port_diversity") or cfg.get("include_dest_concentration"):
            ports = df_sorted.loc[idxs, dest_port_col].fillna(-1).to_numpy(dtype=np.int32)
            port_counts, unique_counts = window_port_stats(ports, window_sizes)
        if dest_ip_col and dest_ip_col in df_sorted.columns:
            dest_ips = df_sorted.loc[idxs, dest_ip_col].fillna("missing").astype(str).to_numpy()
            if cfg.get("include_botnet_context"):
                dest_counts_by_window = window_value_counts(dest_ips, window_sizes)

        if cfg["include_destination_port_count"] and port_counts is not None:
            for window in window_sizes:
                data[f"win_destination_port_count_{window}"][idxs] = port_counts[window].astype("float32")

        if cfg["include_unique_destination_port"] and unique_counts is not None:
            for window in window_sizes:
                data[f"win_unique_destination_port_{window}"][idxs] = unique_counts[window].astype("float32")

        if cfg["include_unique_port_ratio"] and unique_counts is not None and cfg["include_flow_count"]:
            for window in window_sizes:
                unique_vals = unique_counts[window].astype("float32")
                flow_vals = data[f"win_flow_count_{window}"][idxs]
                data[f"win_unique_destination_port_ratio_{window}"][idxs] = unique_vals / np.maximum(flow_vals, 1.0)
                same_port_vals = port_counts[window].astype("float32") if port_counts is not None else np.zeros(group_len, dtype=np.float32)
                data[f"win_same_destination_port_ratio_{window}"][idxs] = same_port_vals / np.maximum(flow_vals, 1.0)
                data[f"win_new_destination_port_indicator_{window}"][idxs] = (same_port_vals <= 1.0).astype("float32")

        if cfg["include_packet_byte_sums"] and packet_total is not None and byte_total is not None:
            packet_group = packet_total[idxs]
            byte_group = byte_total[idxs]
            for window in window_sizes:
                data[f"win_packets_sum_{window}"][idxs] = rolling_sum(packet_group, window)
                data[f"win_bytes_sum_{window}"][idxs] = rolling_sum(byte_group, window)

        if cfg["include_flag_ratios"] and syn_arr is not None and ack_arr is not None and rst_arr is not None and fin_arr is not None:
            syn_group = syn_arr[idxs]
            ack_group = ack_arr[idxs]
            rst_group = rst_arr[idxs]
            fin_group = fin_arr[idxs]
            for window in window_sizes:
                syn_sum = rolling_sum(syn_group, window)
                ack_sum = rolling_sum(ack_group, window)
                rst_sum = rolling_sum(rst_group, window)
                fin_sum = rolling_sum(fin_group, window)
                denom = ack_sum + eps
                data[f"win_syn_ratio_{window}"][idxs] = syn_sum / denom
                data[f"win_rst_ratio_{window}"][idxs] = rst_sum / denom
                data[f"win_fin_ratio_{window}"][idxs] = fin_sum / denom

        if cfg.get("include_behavior_proxies", True):
            if ports is None:
                ports = df_sorted.loc[idxs, dest_port_col].fillna(-1).to_numpy(dtype=np.int32)
            port_switch = np.zeros(group_len, dtype=np.float32)
            if group_len > 1:
                port_switch[1:] = (ports[1:] != ports[:-1]).astype("float32")

            if packet_total is not None:
                packet_group = packet_total[idxs]
                short_flow = (packet_group <= float(cfg.get("short_flow_packet_threshold", 6))).astype("float32")
            else:
                short_flow = np.zeros(group_len, dtype=np.float32)

            if byte_total is not None:
                byte_group = byte_total[idxs]
                small_bytes = (byte_group <= float(cfg.get("small_flow_byte_threshold", 512))).astype("float32")
            else:
                small_bytes = np.zeros(group_len, dtype=np.float32)

            if syn_arr is not None and ack_arr is not None and rst_arr is not None:
                syn_dominant = (syn_arr[idxs] > ack_arr[idxs]).astype("float32")
                rst_dominant = (rst_arr[idxs] > ack_arr[idxs]).astype("float32")
            else:
                syn_dominant = np.zeros(group_len, dtype=np.float32)
                rst_dominant = np.zeros(group_len, dtype=np.float32)

            for window in window_sizes:
                flow_vals = data[f"win_flow_count_{window}"][idxs]
                data[f"win_port_switch_ratio_{window}"][idxs] = rolling_binary_ratio(port_switch, window, flow_vals)
                data[f"win_short_flow_ratio_{window}"][idxs] = rolling_binary_ratio(short_flow, window, flow_vals)
                data[f"win_small_bytes_ratio_{window}"][idxs] = rolling_binary_ratio(small_bytes, window, flow_vals)
                data[f"win_syn_dominant_ratio_{window}"][idxs] = rolling_binary_ratio(syn_dominant, window, flow_vals)
                data[f"win_rst_dominant_ratio_{window}"][idxs] = rolling_binary_ratio(rst_dominant, window, flow_vals)

        if cfg.get("include_periodicity") and timestamp_col and timestamp_col in df_sorted.columns:
            times = to_unix_seconds(df_sorted.loc[idxs, timestamp_col])
            quick_arrival = np.zeros(group_len, dtype=np.float32)
            if group_len > 1:
                quick_arrival[1:] = ((times[1:] - times[:-1]) <= burst_gap_seconds).astype("float32")
            for window in window_sizes:
                data[f"win_burst_count_{window}"][idxs] = rolling_sum(quick_arrival, window)

        if cfg.get("include_botnet_context") and dest_counts_by_window is not None:
            for window in window_sizes:
                flow_vals = data[f"win_flow_count_{window}"][idxs]
                data[f"win_dest_repeat_ratio_{window}"][idxs] = (
                    dest_counts_by_window[window].astype("float32") / np.maximum(flow_vals, 1.0)
                )
                if source_ports is not None and ports is not None:
                    source_counts = window_value_counts(source_ports, [window])[window].astype("float32")
                    pair_values = np.asarray([f"{sp}:{dp}" for sp, dp in zip(source_ports, ports, strict=False)])
                    pair_counts = window_value_counts(pair_values, [window])[window].astype("float32")
                    data[f"win_source_port_repeat_ratio_{window}"][idxs] = source_counts / np.maximum(flow_vals, 1.0)
                    data[f"win_source_dest_port_pair_repeat_ratio_{window}"][idxs] = pair_counts / np.maximum(flow_vals, 1.0)

        if cfg.get("include_low_slow"):
            if packet_total is not None:
                packet_group = packet_total[idxs]
                short_flow = (packet_group <= float(cfg.get("short_flow_packet_threshold", 6))).astype("float32")
            else:
                short_flow = np.zeros(group_len, dtype=np.float32)
            if byte_total is not None:
                byte_group = byte_total[idxs]
                small_bytes = (byte_group <= float(cfg.get("small_flow_byte_threshold", 512))).astype("float32")
            else:
                small_bytes = np.zeros(group_len, dtype=np.float32)
            low_slow = (short_flow * small_bytes).astype("float32")
            for window in window_sizes:
                data[f"win_low_slow_repeat_count_{window}"][idxs] = rolling_sum(low_slow, window)

        if (
            cfg.get("include_port_diversity")
            or cfg.get("include_timing_regularity")
            or cfg.get("include_dest_concentration")
            or cfg.get("include_beaconing_detection")
        ):
            seq_ports = np.zeros(group_len, dtype=np.float32)
            if group_len > 1 and ports is not None:
                seq_ports[1:] = (ports[1:] == ports[:-1] + 1).astype(np.float32)
            high_ports = (ports >= 1024).astype(np.float32) if ports is not None else np.zeros(group_len, dtype=np.float32)

            times = to_unix_seconds(df_sorted.loc[idxs, timestamp_col]) if timestamp_col and timestamp_col in df_sorted.columns else np.zeros(group_len)
            iats = np.zeros(group_len, dtype=np.float32)
            if group_len > 1:
                iats[1:] = times[1:] - times[:-1]

            for window in window_sizes:
                flow_vals = data[f"win_flow_count_{window}"][idxs]

                if cfg.get("include_port_diversity"):
                    data[f"win_sequential_port_ratio_{window}"][idxs] = rolling_sum(seq_ports, window) / np.maximum(flow_vals, 1.0)
                    data[f"win_high_port_ratio_{window}"][idxs] = rolling_sum(high_ports, window) / np.maximum(flow_vals, 1.0)

                if cfg.get("include_timing_regularity") or cfg.get("include_beaconing_detection"):
                    iat_sum = rolling_sum(iats, window)
                    iat_sq_sum = rolling_sum(iats**2, window)
                    mean_iat = iat_sum / np.maximum(flow_vals, 1.0)
                    var_iat = np.maximum(0.0, (iat_sq_sum / np.maximum(flow_vals, 1.0)) - (mean_iat**2))
                    cv = np.sqrt(var_iat) / np.maximum(mean_iat, 1e-6)
                    cv[flow_vals <= 1] = 0.0
                    regularity = 1.0 / (1.0 + cv)
                    if cfg.get("include_timing_regularity"):
                        data[f"win_inter_arrival_cv_{window}"][idxs] = cv
                        data[f"win_inter_arrival_regularity_{window}"][idxs] = regularity
                    if cfg.get("include_beaconing_detection"):
                        data[f"win_beaconing_score_{window}"][idxs] = regularity * (flow_vals / max(float(window), 1.0))

            if cfg.get("include_port_diversity") or cfg.get("include_dest_concentration"):
                for window in window_sizes:
                    p_ent = np.zeros(group_len, dtype=np.float32)
                    p_con = np.zeros(group_len, dtype=np.float32)
                    d_ent = np.zeros(group_len, dtype=np.float32)
                    d_sng = np.zeros(group_len, dtype=np.float32)
                    p_per = np.zeros(group_len, dtype=np.float32)

                    queue = deque()
                    p_c = {}
                    d_c = {}
                    s_p = 0.0
                    s_d = 0.0

                    for i in range(group_len):
                        queue.append(i)
                        if ports is not None:
                            p = int(ports[i])
                            op = p_c.get(p, 0)
                            p_c[p] = op + 1
                            if op + 1 < MAX_Q: s_p += _F_LOG_F_CACHE[op + 1] - _F_LOG_F_CACHE[op]
                        if dest_ips is not None:
                            d = dest_ips[i]
                            od = d_c.get(d, 0)
                            d_c[d] = od + 1
                            if od + 1 < MAX_Q: s_d += _F_LOG_F_CACHE[od + 1] - _F_LOG_F_CACHE[od]

                        if len(queue) > window:
                            old = queue.popleft()
                            if ports is not None:
                                p = int(ports[old])
                                op = p_c[p]
                                p_c[p] = op - 1
                                if op < MAX_Q: s_p += _F_LOG_F_CACHE[op - 1] - _F_LOG_F_CACHE[op]
                                if p_c[p] == 0: del p_c[p]
                            if dest_ips is not None:
                                d = dest_ips[old]
                                od = d_c[d]
                                d_c[d] = od - 1
                                if od < MAX_Q: s_d += _F_LOG_F_CACHE[od - 1] - _F_LOG_F_CACHE[od]
                                if d_c[d] == 0: del d_c[d]

                        count = len(queue)
                        if cfg.get("include_port_diversity") and ports is not None:
                            p_ent[i] = entropy(count, s_p)
                            p_con[i] = 1.0 - float(len(p_c)) / max(count, 1.0)
                        if cfg.get("include_dest_concentration") and dest_ips is not None:
                            d_ent[i] = entropy(count, s_d)
                            d_sng[i] = max(d_c.values()) / max(count, 1.0) if d_c else 0.0
                            p_per[i] = float(len(p_c)) / max(float(len(d_c)), 1.0)

                    if cfg.get("include_port_diversity"):
                        data[f"win_port_entropy_{window}"][idxs] = p_ent
                        data[f"win_port_concentration_{window}"][idxs] = p_con
                    if cfg.get("include_dest_concentration"):
                        data[f"win_dest_ip_entropy_{window}"][idxs] = d_ent
                        data[f"win_single_dest_ratio_{window}"][idxs] = d_sng
                        data[f"win_port_per_dest_ip_{window}"][idxs] = p_per

        if time_window_seconds and ports is not None and source_ports is not None and dest_ips is not None and timestamp_col and timestamp_col in df_sorted.columns:
            times = to_unix_seconds(df_sorted.loc[idxs, timestamp_col])
            packet_group = packet_total[idxs] if packet_total is not None else None
            byte_group = byte_total[idxs] if byte_total is not None else None
            syn_group = syn_arr[idxs] if syn_arr is not None else None
            ack_group = ack_arr[idxs] if ack_arr is not None else None
            rst_group = rst_arr[idxs] if rst_arr is not None else None
            fin_group = fin_arr[idxs] if fin_arr is not None else None
            time_stats = time_window_stats(
                times=times,
                ports=ports,
                source_ports=source_ports,
                dest_ips=dest_ips,
                packet_total=packet_group,
                byte_total=byte_group,
                syn_group=syn_group,
                ack_group=ack_group,
                rst_group=rst_group,
                fin_group=fin_group,
                window_seconds=time_window_seconds,
                include_port_diversity=bool(cfg.get("include_port_diversity")),
                include_timing_regularity=bool(cfg.get("include_timing_regularity")),
                include_dest_concentration=bool(cfg.get("include_dest_concentration")),
                include_beaconing_detection=bool(cfg.get("include_beaconing_detection")),
                short_flow_packet_threshold=float(cfg.get("short_flow_packet_threshold", 6)),
                small_flow_byte_threshold=float(cfg.get("small_flow_byte_threshold", 512)),
                burst_gap_seconds=burst_gap_seconds,
            )
            for seconds, row in time_stats.items():
                data[f"time_flow_count_{seconds}s"][idxs] = row["flow_count"]
                if cfg["include_unique_destination_port"]:
                    data[f"time_unique_destination_port_{seconds}s"][idxs] = row["unique_port"]
                if cfg["include_destination_port_count"]:
                    data[f"time_destination_port_count_{seconds}s"][idxs] = row["port_count"]
                if cfg["include_unique_destination_ip"]:
                    data[f"time_unique_destination_ip_{seconds}s"][idxs] = row["unique_dest"]
                if cfg["include_destination_ip_count"]:
                    data[f"time_destination_ip_count_{seconds}s"][idxs] = row["dest_count"]
                if cfg["include_destination_service_count"]:
                    data[f"time_destination_service_count_{seconds}s"][idxs] = row["service_count"]
                if cfg["include_unique_port_ratio"] and cfg["include_unique_destination_port"]:
                    data[f"time_unique_destination_port_ratio_{seconds}s"][idxs] = row["unique_port"] / np.maximum(row["flow_count"], 1.0)
                if cfg["include_packet_byte_sums"]:
                    data[f"time_packets_sum_{seconds}s"][idxs] = row["packets_sum"]
                    data[f"time_bytes_sum_{seconds}s"][idxs] = row["bytes_sum"]
                if cfg["include_flag_ratios"]:
                    data[f"time_syn_ratio_{seconds}s"][idxs] = row["syn_ratio"]
                    data[f"time_rst_ratio_{seconds}s"][idxs] = row["rst_ratio"]
                    data[f"time_fin_ratio_{seconds}s"][idxs] = row["fin_ratio"]
                if cfg.get("include_behavior_proxies", True):
                    data[f"time_port_switch_ratio_{seconds}s"][idxs] = row["port_switch_ratio"]
                    data[f"time_short_flow_ratio_{seconds}s"][idxs] = row["short_flow_ratio"]
                    data[f"time_small_bytes_ratio_{seconds}s"][idxs] = row["small_bytes_ratio"]
                    data[f"time_syn_dominant_ratio_{seconds}s"][idxs] = row["syn_dom_ratio"]
                    data[f"time_rst_dominant_ratio_{seconds}s"][idxs] = row["rst_dom_ratio"]
                if cfg.get("include_periodicity"):
                    data[f"time_burst_count_{seconds}s"][idxs] = row["burst_count"]
                if cfg.get("include_botnet_context"):
                    data[f"time_dest_repeat_ratio_{seconds}s"][idxs] = row["dest_repeat_ratio"]
                    data[f"time_source_port_repeat_ratio_{seconds}s"][idxs] = row["source_port_repeat_ratio"]
                    data[f"time_source_dest_port_pair_repeat_ratio_{seconds}s"][idxs] = row["source_dest_port_pair_repeat_ratio"]
                if cfg.get("include_low_slow"):
                    data[f"time_low_slow_repeat_count_{seconds}s"][idxs] = row["low_slow_repeat_count"]
                if cfg.get("include_port_diversity"):
                    data[f"time_port_entropy_{seconds}s"][idxs] = row["port_entropy"]
                    data[f"time_port_concentration_{seconds}s"][idxs] = row["port_concentration"]
                    data[f"time_sequential_port_ratio_{seconds}s"][idxs] = row["sequential_port_ratio"]
                    data[f"time_high_port_ratio_{seconds}s"][idxs] = row["high_port_ratio"]
                if cfg.get("include_timing_regularity"):
                    data[f"time_inter_arrival_cv_{seconds}s"][idxs] = row["inter_arrival_cv"]
                    data[f"time_inter_arrival_regularity_{seconds}s"][idxs] = row["inter_arrival_regularity"]
                if cfg.get("include_dest_concentration"):
                    data[f"time_dest_ip_entropy_{seconds}s"][idxs] = row["dest_ip_entropy"]
                    data[f"time_single_dest_ratio_{seconds}s"][idxs] = row["single_dest_ratio"]
                    data[f"time_port_per_dest_ip_{seconds}s"][idxs] = row["port_per_dest_ip"]
                if cfg.get("include_beaconing_detection"):
                    data[f"time_dest_ip_concentration_{seconds}s"][idxs] = row["dest_ip_concentration"]

    feature_frame = pd.DataFrame(data, index=df_sorted.index)
    df_sorted = pd.concat([df_sorted, feature_frame], axis=1)

    if temp_timestamp_col is not None and temp_timestamp_col in df_sorted.columns:
        df_sorted = df_sorted.drop(columns=[temp_timestamp_col])

    if original_row_ids is not None:
        df_sorted = df_sorted.set_index("row_id", drop=False).loc[original_row_ids]
    return df_sorted, new_columns
