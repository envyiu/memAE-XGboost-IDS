"""Time-based sliding window feature computation."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from src.features.window.rolling import MAX_Q, entropy, get_f_log_f


def time_window_stats(
    times: np.ndarray,
    ports: np.ndarray,
    source_ports: np.ndarray,
    dest_ips: np.ndarray,
    packet_total: np.ndarray | None,
    byte_total: np.ndarray | None,
    syn_group: np.ndarray | None,
    ack_group: np.ndarray | None,
    rst_group: np.ndarray | None,
    fin_group: np.ndarray | None,
    window_seconds: list[int],
    include_port_diversity: bool,
    include_timing_regularity: bool,
    include_dest_concentration: bool,
    short_flow_packet_threshold: float,
    small_flow_byte_threshold: float,
    burst_gap_seconds: float,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute all time-based window features for a single source group.

    Returns a dict keyed by window duration (seconds), each containing
    a dict of feature name -> numpy array.
    """
    _F_LOG_F_CACHE = get_f_log_f()
    n = len(times)
    out: dict[int, dict[str, np.ndarray]] = {}

    # --- Pre-compute per-flow binary indicators ---
    short_flow = (
        (packet_total <= short_flow_packet_threshold).astype(np.float32)
        if packet_total is not None else np.zeros(n, dtype=np.float32)
    )
    small_bytes = (
        (byte_total <= small_flow_byte_threshold).astype(np.float32)
        if byte_total is not None else np.zeros(n, dtype=np.float32)
    )
    syn_dominant = (
        (syn_group > ack_group).astype(np.float32)
        if syn_group is not None and ack_group is not None
        else np.zeros(n, dtype=np.float32)
    )
    rst_dominant = (
        (rst_group > ack_group).astype(np.float32)
        if rst_group is not None and ack_group is not None
        else np.zeros(n, dtype=np.float32)
    )
    port_switch = np.zeros(n, dtype=np.float32)
    quick_arrival = np.zeros(n, dtype=np.float32)
    iats = np.zeros(n, dtype=np.float32)
    seq_ports = np.zeros(n, dtype=np.float32)
    high_ports = (ports >= 1024).astype(np.float32)
    low_slow = (short_flow * small_bytes).astype(np.float32)

    if n > 1:
        port_switch[1:] = (ports[1:] != ports[:-1]).astype(np.float32)
        quick_arrival[1:] = ((times[1:] - times[:-1]) <= burst_gap_seconds).astype(np.float32)
        iats[1:] = times[1:] - times[:-1]
        seq_ports[1:] = (ports[1:] == ports[:-1] + 1).astype(np.float32)

    # --- Sliding window loop per time budget ---
    for seconds in window_seconds:
        # Output arrays
        flow_count = np.zeros(n, dtype=np.float32)
        unique_port = np.zeros(n, dtype=np.float32)
        port_count = np.zeros(n, dtype=np.float32)
        unique_dest = np.zeros(n, dtype=np.float32)
        dest_count = np.zeros(n, dtype=np.float32)
        service_count = np.zeros(n, dtype=np.float32)
        source_port_repeat_ratio = np.zeros(n, dtype=np.float32)
        source_dest_port_pair_repeat_ratio = np.zeros(n, dtype=np.float32)
        packets_sum = np.zeros(n, dtype=np.float32)
        bytes_sum = np.zeros(n, dtype=np.float32)
        syn_ratio = np.zeros(n, dtype=np.float32)
        rst_ratio = np.zeros(n, dtype=np.float32)
        fin_ratio = np.zeros(n, dtype=np.float32)
        port_switch_ratio = np.zeros(n, dtype=np.float32)
        short_flow_ratio = np.zeros(n, dtype=np.float32)
        small_bytes_ratio = np.zeros(n, dtype=np.float32)
        syn_dom_ratio = np.zeros(n, dtype=np.float32)
        rst_dom_ratio = np.zeros(n, dtype=np.float32)
        burst_count_arr = np.zeros(n, dtype=np.float32)
        dest_repeat_ratio = np.zeros(n, dtype=np.float32)
        low_slow_repeat_count = np.zeros(n, dtype=np.float32)

        port_entropy_arr = np.zeros(n, dtype=np.float32)
        port_concentration_arr = np.zeros(n, dtype=np.float32)
        sequential_port_ratio_arr = np.zeros(n, dtype=np.float32)
        high_port_ratio_arr = np.zeros(n, dtype=np.float32)
        inter_arrival_cv_arr = np.zeros(n, dtype=np.float32)
        inter_arrival_regularity_arr = np.zeros(n, dtype=np.float32)
        dest_ip_entropy_arr = np.zeros(n, dtype=np.float32)
        single_dest_ratio_arr = np.zeros(n, dtype=np.float32)
        port_per_dest_ip_arr = np.zeros(n, dtype=np.float32)

        # Sliding window state
        queue: deque[int] = deque()
        port_counter: dict[int, int] = {}
        dest_counter: dict[str, int] = {}
        service_counter: dict[tuple[str, int], int] = {}
        source_port_counter: dict[int, int] = {}
        source_dest_port_counter: dict[tuple[int, int], int] = {}

        packet_running = 0.0
        byte_running = 0.0
        syn_running = 0.0
        ack_running = 0.0
        rst_running = 0.0
        fin_running = 0.0
        port_switch_running = 0.0
        short_running = 0.0
        small_running = 0.0
        syn_dom_running = 0.0
        rst_dom_running = 0.0
        burst_running = 0.0
        low_slow_running = 0.0

        sum_port_f_log_f = 0.0
        sum_dest_f_log_f = 0.0
        seq_port_running = 0.0
        high_port_running = 0.0
        iat_sum = 0.0
        iat_sq_sum = 0.0

        for i in range(n):
            queue.append(i)

            # Add current flow to counters
            p = int(ports[i])
            old_p_c = port_counter.get(p, 0)
            port_counter[p] = old_p_c + 1
            if old_p_c + 1 < MAX_Q:
                sum_port_f_log_f += _F_LOG_F_CACHE[old_p_c + 1] - _F_LOG_F_CACHE[old_p_c]

            d = dest_ips[i]
            old_d_c = dest_counter.get(d, 0)
            dest_counter[d] = old_d_c + 1
            if old_d_c + 1 < MAX_Q:
                sum_dest_f_log_f += _F_LOG_F_CACHE[old_d_c + 1] - _F_LOG_F_CACHE[old_d_c]

            key = (d, p)
            service_counter[key] = service_counter.get(key, 0) + 1
            source_port_counter[source_ports[i]] = source_port_counter.get(source_ports[i], 0) + 1
            port_pair_key = (int(source_ports[i]), p)
            source_dest_port_counter[port_pair_key] = source_dest_port_counter.get(port_pair_key, 0) + 1

            seq_port_running += float(seq_ports[i])
            high_port_running += float(high_ports[i])
            iat_sum += float(iats[i])
            iat_sq_sum += float(iats[i] ** 2)

            if packet_total is not None:
                packet_running += float(packet_total[i])
            if byte_total is not None:
                byte_running += float(byte_total[i])
            if syn_group is not None:
                syn_running += float(syn_group[i])
            if ack_group is not None:
                ack_running += float(ack_group[i])
            if rst_group is not None:
                rst_running += float(rst_group[i])
            if fin_group is not None:
                fin_running += float(fin_group[i])
            port_switch_running += float(port_switch[i])
            short_running += float(short_flow[i])
            small_running += float(small_bytes[i])
            syn_dom_running += float(syn_dominant[i])
            rst_dom_running += float(rst_dominant[i])
            burst_running += float(quick_arrival[i])
            low_slow_running += float(low_slow[i])

            # Evict expired flows
            while queue and (times[i] - times[queue[0]]) > seconds:
                old = queue.popleft()
                old_port = int(ports[old])
                old_dest = dest_ips[old]
                old_key = (old_dest, old_port)

                old_p_c = port_counter[old_port]
                port_counter[old_port] -= 1
                if old_p_c < MAX_Q:
                    sum_port_f_log_f += _F_LOG_F_CACHE[old_p_c - 1] - _F_LOG_F_CACHE[old_p_c]
                if port_counter[old_port] <= 0:
                    port_counter.pop(old_port, None)

                old_d_c = dest_counter[old_dest]
                dest_counter[old_dest] -= 1
                if old_d_c < MAX_Q:
                    sum_dest_f_log_f += _F_LOG_F_CACHE[old_d_c - 1] - _F_LOG_F_CACHE[old_d_c]
                if dest_counter[old_dest] <= 0:
                    dest_counter.pop(old_dest, None)

                service_counter[old_key] -= 1
                if service_counter[old_key] <= 0:
                    service_counter.pop(old_key, None)
                old_source_port = source_ports[old]
                source_port_counter[old_source_port] -= 1
                if source_port_counter[old_source_port] <= 0:
                    source_port_counter.pop(old_source_port, None)
                old_pair_key = (int(old_source_port), old_port)
                source_dest_port_counter[old_pair_key] -= 1
                if source_dest_port_counter[old_pair_key] <= 0:
                    source_dest_port_counter.pop(old_pair_key, None)

                seq_port_running -= float(seq_ports[old])
                high_port_running -= float(high_ports[old])
                iat_sum -= float(iats[old])
                iat_sq_sum -= float(iats[old] ** 2)

                if packet_total is not None:
                    packet_running -= float(packet_total[old])
                if byte_total is not None:
                    byte_running -= float(byte_total[old])
                if syn_group is not None:
                    syn_running -= float(syn_group[old])
                if ack_group is not None:
                    ack_running -= float(ack_group[old])
                if rst_group is not None:
                    rst_running -= float(rst_group[old])
                if fin_group is not None:
                    fin_running -= float(fin_group[old])
                port_switch_running -= float(port_switch[old])
                short_running -= float(short_flow[old])
                small_running -= float(small_bytes[old])
                syn_dom_running -= float(syn_dominant[old])
                rst_dom_running -= float(rst_dominant[old])
                burst_running -= float(quick_arrival[old])
                low_slow_running -= float(low_slow[old])

            # Record features for flow i
            count = float(len(queue))
            flow_count[i] = count
            unique_port[i] = float(len(port_counter))
            port_count[i] = float(port_counter.get(ports[i], 0))
            unique_dest[i] = float(len(dest_counter))
            dest_count[i] = float(dest_counter.get(dest_ips[i], 0))
            service_count[i] = float(service_counter.get(key, 0))
            dest_repeat_ratio[i] = float(dest_counter.get(dest_ips[i], 0) / max(count, 1.0))
            source_port_repeat_ratio[i] = float(source_port_counter.get(source_ports[i], 0) / max(count, 1.0))
            source_dest_port_pair_repeat_ratio[i] = float(source_dest_port_counter.get(port_pair_key, 0) / max(count, 1.0))
            packets_sum[i] = float(packet_running)
            bytes_sum[i] = float(byte_running)
            denom = max(ack_running, 1e-6)
            syn_ratio[i] = float(syn_running / denom)
            rst_ratio[i] = float(rst_running / denom)
            fin_ratio[i] = float(fin_running / denom)
            port_switch_ratio[i] = float(port_switch_running / max(count, 1.0))
            short_flow_ratio[i] = float(short_running / max(count, 1.0))
            small_bytes_ratio[i] = float(small_running / max(count, 1.0))
            syn_dom_ratio[i] = float(syn_dom_running / max(count, 1.0))
            rst_dom_ratio[i] = float(rst_dom_running / max(count, 1.0))
            burst_count_arr[i] = float(burst_running)
            low_slow_repeat_count[i] = float(low_slow_running)

            if include_port_diversity:
                port_entropy_arr[i] = entropy(count, sum_port_f_log_f)
                port_concentration_arr[i] = 1.0 - (float(len(port_counter)) / max(count, 1.0))
                sequential_port_ratio_arr[i] = seq_port_running / max(count, 1.0)
                high_port_ratio_arr[i] = high_port_running / max(count, 1.0)

            if include_timing_regularity:
                if count > 1:
                    mean_iat = iat_sum / count
                    var_iat = max(0.0, (iat_sq_sum / count) - (mean_iat ** 2))
                    cv = math.sqrt(var_iat) / max(mean_iat, 1e-6)
                    inter_arrival_cv_arr[i] = cv
                    inter_arrival_regularity_arr[i] = 1.0 / (1.0 + cv)
                else:
                    inter_arrival_cv_arr[i] = 0.0
                    inter_arrival_regularity_arr[i] = 1.0

            if include_dest_concentration:
                dest_ip_entropy_arr[i] = entropy(count, sum_dest_f_log_f)
                max_freq = max(dest_counter.values()) if dest_counter else 0
                single_dest_ratio_arr[i] = float(max_freq) / max(count, 1.0)
                port_per_dest_ip_arr[i] = float(len(port_counter)) / max(float(len(dest_counter)), 1.0)

        out[seconds] = {
            "flow_count": flow_count,
            "unique_port": unique_port,
            "port_count": port_count,
            "unique_dest": unique_dest,
            "dest_count": dest_count,
            "service_count": service_count,
            "source_port_repeat_ratio": source_port_repeat_ratio,
            "source_dest_port_pair_repeat_ratio": source_dest_port_pair_repeat_ratio,
            "packets_sum": packets_sum,
            "bytes_sum": bytes_sum,
            "syn_ratio": syn_ratio,
            "rst_ratio": rst_ratio,
            "fin_ratio": fin_ratio,
            "port_switch_ratio": port_switch_ratio,
            "short_flow_ratio": short_flow_ratio,
            "small_bytes_ratio": small_bytes_ratio,
            "syn_dom_ratio": syn_dom_ratio,
            "rst_dom_ratio": rst_dom_ratio,
            "burst_count": burst_count_arr,
            "dest_repeat_ratio": dest_repeat_ratio,
            "low_slow_repeat_count": low_slow_repeat_count,
            "port_entropy": port_entropy_arr,
            "port_concentration": port_concentration_arr,
            "sequential_port_ratio": sequential_port_ratio_arr,
            "high_port_ratio": high_port_ratio_arr,
            "inter_arrival_cv": inter_arrival_cv_arr,
            "inter_arrival_regularity": inter_arrival_regularity_arr,
            "dest_ip_entropy": dest_ip_entropy_arr,
            "single_dest_ratio": single_dest_ratio_arr,
            "port_per_dest_ip": port_per_dest_ip_arr,
        }
    return out
