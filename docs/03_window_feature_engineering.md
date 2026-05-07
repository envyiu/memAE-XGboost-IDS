# Module: Window Feature Engineering

File này đi sâu vào module `src/features/window/`, tức lớp feature engineering theo ngữ cảnh dòng chảy mạng. Đây là phần biến từng flow độc lập của CIC-IDS2017 thành flow có bối cảnh theo host, theo cổng, theo đích và theo thời gian. Trong recipe chính `zdr5`, module này được bật qua `configs/window_features_zdr5.yaml` và được gọi từ `src/preprocessing/run_preprocessing.py`.

---

## 1. Vai trò trong pipeline

Window features được tính trước khi fit/transform bằng `IDSPreprocessor`.

Luồng thực tế:

```text
clean parquet
  -> load theo từng source_file
  -> add_window_features()
  -> ghép feature gốc + window feature
  -> fit/transform IDSPreprocessor
  -> X_*.npy
```

Điểm quan trọng là window features vẫn đi qua cùng bước clipping, median impute và standardize như feature số gốc. Vì vậy giá trị raw như `win_bytes_sum_1000` có thể rất lớn nhưng khi đến model đã được chuẩn hóa.

Trong config chính:

```yaml
enabled: true
window_scope: full_source_file
group_by: [source_file, source_ip]
order_by: [timestamp, row_id]
window_sizes: [10, 50, 200, 1000]
time_window_seconds: [60, 300, 600, 3600]
```

Nghĩa là mỗi source host trong từng file gốc có một chuỗi flow riêng, được sắp theo thời gian rồi theo `row_id`, sau đó tính thống kê quá khứ/tới hiện tại trong cửa sổ theo số flow và theo giây.

---

## 2. Các file chính

```text
src/features/window/config.py       # default config, merge, resolve cột phụ thuộc
src/features/window/names.py        # sinh danh sách tên feature theo config
src/features/window/rolling.py      # primitive rolling count/sum/entropy
src/features/window/time_window.py  # sliding window theo thời gian
src/features/window/engine.py       # add_window_features(), orchestration chính
```

`engine.py` không tự quyết định config từ file YAML. Nó nhận dict config đã được đọc bởi caller. `run_preprocessing.py` là nơi đọc YAML, resolve config theo `column_schema.json`, rồi truyền vào `add_window_features()`.

---

## 3. Config resolution

### 3.1 `merge_config()`

`merge_config(config)` phủ user config lên `DEFAULT_WINDOW_CONFIG`.

Với key dạng dict như `flag_columns`, `packet_columns`, `byte_columns`, hàm merge theo từng key con thay vì thay toàn bộ dict. Điều này cho phép config chỉ override một cột trong nhóm mà vẫn giữ default cho các cột còn lại.

### 3.2 `resolve_window_config()`

`resolve_window_config(config, available_columns)` xử lý trường hợp dataset thiếu cột ngữ cảnh.

Logic:

```text
Nếu destination_port_column tồn tại:
  effective_context_key_column = destination_port
  effective_context_key_source = destination_port

Nếu thiếu destination_port nhưng có fallback_context_key_column:
  destination_port_column = fallback
  include_service_context = false

Nếu không có cả hai:
  effective_context_key_column = null
  include_service_context = false
  include_unique_destination_port = false
  include_destination_port_count = false
  include_unique_port_ratio = false
```

Trong CIC-IDS2017 hiện tại, `destination_port` có tồn tại nên recipe chính giữ được đầy đủ context theo service port.

### 3.3 `window_required_columns()`

Hàm này trả về danh sách cột phải load thêm từ parquet để tính window. Nó lấy:

- `group_by` và `order_by`: ví dụ `source_file`, `source_ip`, `timestamp`, `row_id`.
- Cột port/IP nếu bật feature tương ứng.
- Cột packet/byte nếu bật `include_packet_byte_sums`.
- Cột flag nếu bật `include_flag_ratios`.

`run_preprocessing.py` chỉ load `row_id`, `attack_family`, numerical features, và các cột required này. Đây là điểm kiểm soát RAM vì không cần load toàn bộ parquet vào từng pass.

---

## 4. Cơ chế sort và giữ nguyên thứ tự row

`add_window_features(df, config)` luôn làm việc theo thứ tự causal trong từng group:

1. Copy dataframe đầu vào.
2. Parse `timestamp` sang cột tạm `__timestamp_for_sort`.
3. Sort theo `group_by + order_by`.
4. Tính toàn bộ window features trên dataframe đã sort.
5. Nếu có `row_id`, set index theo `row_id` và trả dataframe về đúng thứ tự row ban đầu.

Điều này giải quyết hai nhu cầu trái nhau:

- Feature phải được tính theo trình tự thời gian đúng.
- Output phải giữ đúng thứ tự với split CSV để ghi vào `X_*.npy`.

Nếu timestamp parse fail hoàn toàn với `dayfirst=True`, code thử lại `dayfirst=False`. Nếu dùng time-window mà timestamp vẫn có NaN, `to_unix_seconds()` sẽ forward fill/backward fill; nếu vẫn thiếu thì fallback bằng dãy giây nhân tạo theo vị trí.

---

## 5. Nhóm feature tĩnh theo service context

Khi `include_service_context: true`, module tạo các feature không cần window:

```text
ctx_dest_port_is_system       # destination_port < 1024
ctx_dest_port_is_registered   # 1024 <= destination_port < 49152
ctx_dest_port_is_dynamic      # destination_port >= 49152
ctx_service_auth
ctx_service_database
ctx_service_fileshare
ctx_service_infra
ctx_service_web
```

Các family port lấy từ config:

```yaml
auth: [21, 22, 23, 25, 110, 143, 3389]
web: [80, 443, 8080, 8443]
infra: [53, 67, 68, 123, 161]
fileshare: [135, 137, 138, 139, 445]
database: [1433, 1521, 3306, 5432, 6379]
```

Khi `include_source_port_context: true`, module tạo thêm nhóm tương tự cho `source_port`:

```text
ctx_source_port_is_system
ctx_source_port_is_registered
ctx_source_port_is_dynamic
ctx_source_dest_same_port
ctx_source_service_*
```

Khi `include_watched_ports: true`, mỗi watched port tạo hai indicator:

```text
ctx_source_port_8080
ctx_dest_port_8080
```

Những feature này giúp model biết cổng đang xét thuộc lớp dịch vụ nào mà không phải dùng trực tiếp IP như một feature học máy.

---

## 6. Count-based window features

Count-based window dùng `window_sizes`, ví dụ `[10, 50, 200, 1000]`. Với mỗi flow thứ `i` trong group, window `W` gồm tối đa `W` flow gần nhất tính cả flow hiện tại.

### 6.1 Flow count

```text
win_flow_count_W = min(vị trí trong group + 1, W)
```

Feature này là denominator cho nhiều tỷ lệ khác. Với các flow đầu tiên trong group, denominator nhỏ hơn W.

### 6.2 Port statistics

`window_port_stats()` trả về hai mảng cho từng W:

- `win_destination_port_count_W`: số lần destination port hiện tại xuất hiện trong window.
- `win_unique_destination_port_W`: số destination port khác nhau trong window.

Từ đó tạo:

```text
win_unique_destination_port_ratio_W = unique_port / flow_count
win_same_destination_port_ratio_W = current_port_count / flow_count
win_new_destination_port_indicator_W = 1 nếu current_port_count <= 1
```

Ý nghĩa:

- PortScan thường có `unique_destination_port_ratio` cao và `new_destination_port_indicator` cao.
- DDoS/DoS vào một dịch vụ cụ thể thường có `same_destination_port_ratio` cao.

### 6.3 Packet/byte sums

Nếu bật `include_packet_byte_sums`, module tính:

```text
packet_total = total_fwd_packets + total_backward_packets
byte_total = total_length_of_fwd_packets + total_length_of_bwd_packets

win_packets_sum_W
win_bytes_sum_W
```

Code dùng `rolling_sum()` dựa trên cumulative sum nên O(n) cho mỗi W.

### 6.4 TCP flag ratios

Nếu bật `include_flag_ratios`, module lấy:

```text
syn_flag_count
ack_flag_count
rst_flag_count
fin_flag_count
```

Rồi tính:

```text
win_syn_ratio_W = sum(SYN) / max(sum(ACK), 1e-6)
win_rst_ratio_W = sum(RST) / max(sum(ACK), 1e-6)
win_fin_ratio_W = sum(FIN) / max(sum(ACK), 1e-6)
```

Tỷ lệ này nhấn mạnh bất thường kiểu SYN-dominant hoặc reset-heavy trong chuỗi flow.

### 6.5 Behavior proxies

Khi `include_behavior_proxies: true`, module tạo các chỉ báo nhị phân trước rồi lấy rolling ratio:

```text
port_switch[i] = destination_port[i] != destination_port[i-1]
short_flow[i] = packet_total[i] <= short_flow_packet_threshold
small_bytes[i] = byte_total[i] <= small_flow_byte_threshold
syn_dominant[i] = syn_flag_count[i] > ack_flag_count[i]
rst_dominant[i] = rst_flag_count[i] > ack_flag_count[i]
```

Feature sinh ra:

```text
win_port_switch_ratio_W
win_short_flow_ratio_W
win_small_bytes_ratio_W
win_syn_dominant_ratio_W
win_rst_dominant_ratio_W
```

Trong config chính:

```yaml
short_flow_packet_threshold: 6
small_flow_byte_threshold: 512
```

### 6.6 Periodicity và burst

Khi `include_periodicity: true`, module tính:

```text
quick_arrival[i] = timestamp[i] - timestamp[i-1] <= burst_gap_seconds
win_burst_count_W = rolling_sum(quick_arrival, W)
```

Với `burst_gap_seconds: 1.0`, feature này đo số flow đến sát nhau trong window. Nó hữu ích cho burst traffic, brute force và flooding.

### 6.7 Botnet context

Khi `include_botnet_context: true`, module dùng `window_value_counts()` trên destination IP, source port và cặp source/destination port:

```text
win_dest_repeat_ratio_W
win_source_port_repeat_ratio_W
win_source_dest_port_pair_repeat_ratio_W
```

Ý nghĩa:

- Botnet/callback traffic có thể lặp lại một đích hoặc một pattern cổng.
- Brute force có thể lặp source port/dest port theo chu kỳ.

### 6.8 Low-slow

Khi `include_low_slow: true`:

```text
low_slow = short_flow * small_bytes
win_low_slow_repeat_count_W = rolling_sum(low_slow, W)
```

Feature này không phân biệt nhãn DoS trực tiếp, nhưng nhấn mạnh chuỗi flow nhỏ, ngắn, lặp lại.

### 6.9 Port diversity, timing regularity, destination concentration

Nhóm này dùng thêm entropy và thống kê tập trung:

```text
win_port_entropy_W
win_port_concentration_W
win_sequential_port_ratio_W
win_high_port_ratio_W
win_inter_arrival_cv_W
win_inter_arrival_regularity_W
win_dest_ip_entropy_W
win_single_dest_ratio_W
win_port_per_dest_ip_W
```

Công thức chính:

```text
entropy = log2(count) - sum(freq * log2(freq)) / count
port_concentration = 1 - unique_ports / count
inter_arrival_cv = std(inter_arrival_time) / mean(inter_arrival_time)
inter_arrival_regularity = 1 / (1 + cv)
single_dest_ratio = max(destination_ip_frequency) / count
port_per_dest_ip = unique_ports / unique_dest_ips
```

`rolling.py` có cache `f * log2(f)` tới `MAX_Q = 200000` để giảm chi phí entropy trong loop.

---

## 7. Time-based window features

Time-based window dùng `time_window_seconds`, ví dụ `[60, 300, 600, 3600]`. Với mỗi flow hiện tại, window gồm tất cả flow cùng group có timestamp nằm trong khoảng:

```text
times[current] - times[old] <= seconds
```

`time_window_stats()` dùng queue hai đầu để thêm flow hiện tại và loại flow hết hạn. Với mỗi seconds, nó duy trì state chạy:

- Counter port, destination IP, service `(dest_ip, dest_port)`.
- Counter source port, cặp `(source_port, dest_port)`.
- Running sum packet, byte, flag, burst, low-slow.
- Running entropy accumulator cho port và destination IP.
- Running sum/square sum của inter-arrival time.

Feature sinh ra tương ứng với count-window nhưng có prefix `time_`:

```text
time_flow_count_60s
time_unique_destination_port_60s
time_destination_port_count_60s
time_unique_destination_ip_60s
time_destination_ip_count_60s
time_destination_service_count_60s
time_unique_destination_port_ratio_60s
time_packets_sum_60s
time_bytes_sum_60s
time_syn_ratio_60s
time_rst_ratio_60s
time_fin_ratio_60s
time_port_switch_ratio_60s
time_short_flow_ratio_60s
time_small_bytes_ratio_60s
time_syn_dominant_ratio_60s
time_rst_dominant_ratio_60s
time_burst_count_60s
time_dest_repeat_ratio_60s
time_source_port_repeat_ratio_60s
time_source_dest_port_pair_repeat_ratio_60s
time_low_slow_repeat_count_60s
time_port_entropy_60s
time_port_concentration_60s
time_sequential_port_ratio_60s
time_high_port_ratio_60s
time_inter_arrival_cv_60s
time_inter_arrival_regularity_60s
time_dest_ip_entropy_60s
time_single_dest_ratio_60s
time_port_per_dest_ip_60s
time_dest_ip_concentration_60s
```

Các seconds khác lặp lại cùng cấu trúc tên.

Khi `include_beaconing_detection=true`, count-window cũng sinh thêm:

```text
win_beaconing_score_W = win_inter_arrival_regularity_W * (win_flow_count_W / W)
```

Feature này nhấn mạnh pattern callback đều đặn và đủ dày trong cùng một source group. `time_dest_ip_concentration_Ts` là `max_dest_ip_count / flow_count`, dùng để bắt pattern ít destination IP lặp lại theo thời gian.

---

## 8. Feature naming contract

`window_feature_names(config)` là source of truth cho thứ tự tên feature. `add_window_features()` tạo `data` dict theo đúng danh sách này:

```python
new_columns = window_feature_names(cfg)
data = {name: np.zeros(n_rows, dtype=np.float32) for name in new_columns}
```

Vì vậy nếu thêm feature mới, cần sửa đồng thời:

1. `window_feature_names()` để schema biết tên và thứ tự.
2. `add_window_features()` hoặc `time_window_stats()` để ghi giá trị.
3. Test để đảm bảo tên có trong output và giá trị finite.

Nếu chỉ thêm computation mà quên name, feature không được xuất ra. Nếu thêm name mà quên ghi giá trị, cột sẽ toàn 0 vì dict được khởi tạo bằng zeros.

---

## 9. Window scope trong preprocessing

`preprocess_experiment()` hỗ trợ ba scope:

### 9.1 `full_source_file`

```text
Tính window trên toàn bộ source_file trước, sau đó mới filter row theo split.
```

Đây là default của `zdr5`. Nó mô phỏng trạng thái một IDS đọc luồng theo file nguồn đầy đủ. Ưu điểm là bối cảnh không bị cắt vụn theo split; nhược điểm là các benign/attack cùng source file đều cùng tham gia thống kê context, nên benchmark phải dựa vào host-disjoint split để giảm leak theo entity.

### 9.2 `split_only`

```text
Chỉ tính window trên rows thuộc split hiện tại.
```

Scope này chặt hơn về split isolation nhưng ít giống stream thật, vì flow trước đó ở cùng source file nhưng thuộc split khác bị biến mất.

### 9.3 `causal_past_only`

```text
train: train
val: train + val
test_seen: train + val + test_seen
test_zero_day: train + val + test_zero_day
```

Scope này cố mô phỏng quá khứ causal theo từng phase. Tuy nhiên cần đọc kỹ nếu dùng cho benchmark vì `test_seen` và `test_zero_day` được xử lý tách nhau, không cùng một timeline test duy nhất.

---

## 10. Rủi ro leak và cách module giảm rủi ro

Window features không dùng `label`, `attack_family`, `original_label` để tính feature. Các cột metadata như `source_ip`, `destination_ip`, `timestamp` không nằm trong numerical features gốc nhưng được load để tính ngữ cảnh.

Rủi ro chính:

- `group_by` quá chi tiết hoặc chứa định danh gần nhãn có thể làm model học entity thay vì hành vi.
- `full_source_file` có thể cho context từ rows không thuộc split hiện tại.
- Các cổng dịch vụ quá đặc thù dataset có thể tạo shortcut nếu split không host-disjoint.

Recipe chính giảm rủi ro bằng:

- Split theo host: `group_columns = (source_file, source_ip)`.
- Không đưa IP/timestamp trực tiếp vào `feature_order`.
- Chỉ đưa thống kê hành vi đã aggregate vào model.

---

## 11. Kiểm thử hiện có

`tests/test_pipeline_hardening.py::test_boosted_window_features_are_finite_and_named` tạo dataframe nhỏ có:

- 5 flow cùng `source_file/source_ip`.
- Timestamp cách nhau 500ms.
- Port 80, 443, 8080.
- Packet/byte/flag tối thiểu.

Test kiểm tra:

- `window_feature_names(cfg)` khớp với cột trả về từ `add_window_features()`.
- Các cột nâng cao như `win_dest_repeat_ratio_3`, `time_dest_repeat_ratio_60s`, `win_burst_count_3`, `time_burst_count_60s`, `win_low_slow_repeat_count_3`, `time_low_slow_repeat_count_60s` tồn tại.
- Toàn bộ feature là finite.

Nếu mở rộng module window, nên thêm test tương tự cho nhóm feature mới, đặc biệt là entropy/timing vì dễ phát sinh NaN khi denominator nhỏ.

---

## 12. Checklist khi chỉnh window features

1. Thêm/tắt feature bằng config trước, tránh hard-code trực tiếp trong caller.
2. Nếu feature cần cột mới, cập nhật `window_required_columns()`.
3. Nếu feature có tên mới, cập nhật `window_feature_names()`.
4. Nếu feature dùng timestamp, đảm bảo fallback trong `to_unix_seconds()` không tạo NaN/inf.
5. Giữ output `float32` để memmap không phình kích thước.
6. Chạy test window và ít nhất một preprocess nhỏ trước khi chạy full benchmark.
7. Kiểm tra `data/processed/{experiment}/feature_schema.json` để xác nhận feature_order đúng như mong đợi.
