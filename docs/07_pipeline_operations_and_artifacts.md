# Module: Pipeline Orchestration, Configs & Artifacts

File này mô tả cách toàn bộ dự án được nối lại bởi script chính:

```text
scripts/run_full_pipeline_all_families.py
configs/*.yaml
data/*
artifacts/*
reports/*
tests/test_pipeline_hardening.py
```

Nội dung tập trung vào vận hành pipeline, tên experiment, skip/retrain logic, artifact readiness, và cách đọc cấu trúc output.

---

## 1. Entrypoint chính

Script chính:

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py --families all --force-retrain
```

Script này tự gọi các module:

```text
clean_dataset()
create_leave_one_family_out_split()
preprocess_experiment()
train_memae()
export_features()
train_xgboost_feature_set()
train_score_fusion()
generate_detector_calibration_report()
generate_fusion_calibration_report()
```

Stages được định nghĩa:

```python
STAGES = ("split", "preprocess", "memae", "features", "xgboost", "fusion", "reports")
```

CLI có thể chạy một đoạn pipeline bằng:

```bash
--start-at preprocess --stop-after xgboost
```

`start_at` phải đứng trước hoặc bằng `stop_after`, nếu không script raise `ValueError`.

---

## 2. Family set và low-support families

Family chính:

```python
DEFAULT_FAMILIES = (
  "web_attack",
  "botnet",
  "portscan",
  "ddos",
  "dos",
  "brute_force",
)
```

Family bị loại khỏi benchmark chính:

```python
LOW_SUPPORT_EXCLUDED_FAMILIES = {
  "heartbleed": 11,
  "infiltration": 36,
}
```

Nếu user yêu cầu `heartbleed` hoặc `infiltration` mà không truyền `--allow-low-support`, script dừng với lỗi. Đây là guard để benchmark chính không bị méo bởi support quá nhỏ.

Alias family:

```text
bruteforce   -> brute_force
brute-force  -> brute_force
webattack    -> web_attack
web-attack   -> web_attack
all          -> DEFAULT_FAMILIES
```

---

## 3. Naming convention

### 3.1 Experiment name

```python
experiment = zero_day_{family}_{experiment_suffix}
```

Default:

```text
experiment_suffix = host_disjoint_zdr5
```

Ví dụ:

```text
zero_day_botnet_host_disjoint_zdr5
```

### 3.2 Feature set

```python
feature_set = {experiment}_{variant_suffix}
```

Default:

```text
variant_suffix = targetsel_zdr5
```

Ví dụ:

```text
zero_day_botnet_host_disjoint_zdr5_targetsel_zdr5
```

Feature set là tên dùng chung cho:

```text
artifacts/memae/{feature_set}
data/features/{feature_set}
artifacts/xgboost/{feature_set}
```

### 3.3 Fusion artifact

```python
fusion_artifact = {feature_set}_{fusion_suffix}
```

Default:

```text
fusion_suffix = scorefusion
```

Ví dụ:

```text
zero_day_botnet_host_disjoint_zdr5_targetsel_zdr5_scorefusion
```

### 3.4 Summary suffix

`_summary_suffix()` tạo tên summary suffixed và cũng được dùng trong tên thư mục `reports/runs/{timestamp}_{suffix}`. Nếu chạy một family, prefix family được thêm vào để không ghi nhầm thành summary generic.

Ví dụ test hiện có đảm bảo:

```text
_summary_suffix("targetsel", "host", ["botnet"]) = botnet_host_targetsel
_summary_suffix("targetsel", "host", DEFAULT_FAMILIES) = host_targetsel
```

---

## 4. Split group modes

```python
SPLIT_GROUP_COLUMNS = {
  "exact_flow": ("source_file", "source_ip", "destination_ip", "destination_port"),
  "host": ("source_file", "source_ip"),
}
```

Default:

```bash
--split-group-mode host
```

Host-disjoint là recipe chính. Nó đảm bảo cùng `(source_file, source_ip)` không xuất hiện ở nhiều split. Đây là lựa chọn phù hợp với window config cũng group theo `(source_file, source_ip)`.

`exact_flow` chặt theo flow endpoint hơn nhưng có thể để cùng source host xuất hiện ở nhiều split nếu destination/port khác nhau.

---

## 5. Benchmark mode

CLI default:

```bash
--benchmark-mode host_disjoint_window
```

Nếu không truyền explicit mode, `_benchmark_mode_from_window_config()` đọc window config:

```text
window enabled -> contextual_window
window disabled -> strict_nowindow
```

Trong run chính, explicit default `host_disjoint_window` được ghi vào schema/report. Giá trị này đi vào:

```text
data/processed/{experiment}/feature_schema.json
reports/*summary*.json
```

---

## 6. Config files

### 6.1 `configs/window_features_zdr5.yaml`

Điều khiển feature engineering theo context.

Các nhóm đang bật:

```text
flow_count
destination port/IP counts
service context
source port context
watched ports
packet/byte sums
TCP flag ratios
behavior proxies
periodicity
botnet context
low-slow
count windows: 50, 200, 1000
time windows: 60s, 600s, 3600s
watched_ports: 8080
```

Nếu thay file này, cần chạy lại từ `preprocess` trở đi vì `X_*.npy`, MemAE checkpoint, `F_*.npy`, XGBoost, fusion đều phụ thuộc feature order.

### 6.2 `configs/memae_targeted.yaml`

Điều khiển kiến trúc/training MemAE:

```text
latent_dim=48
memory_size=128
shrink_threshold=0.0078125
epochs=65
batch_size=4096
entropy_weight=0.0002
patience=10
selection=seen_recall_at_benign_fpr @ 1% FPR
```

Nếu thay file này, cần chạy lại từ `memae` trở đi.

### 6.3 `configs/xgboost_zdr5.yaml`

Điều khiển XGBoost và sampling:

```text
objective=binary:logistic
eval_metric=aucpr
n_estimators=1200
early_stopping_rounds=70
learning_rate=0.04
max_depth=7
min_child_weight=2
subsample=0.85
colsample_bytree=0.85
reg_alpha=0.0005
reg_lambda=1.0
family_balance=true
max_train_samples=600000
max_val_samples=150000
max_samples_per_attack_family=80000
benign_to_attack_ratio=2.5
```

Nếu thay file này, cần chạy lại từ `xgboost` trở đi; fusion cũng cần train lại vì fusion dùng score XGBoost.

---

## 7. Skip/retrain logic

Script chỉ chạy stages nếu:

```text
--force-retrain được bật
hoặc artifact cần thiết chưa tồn tại / không tương thích
```

Readiness helpers:

### 7.1 `_processed_ready(experiment)`

Yêu cầu đủ:

```text
X_train.npy
X_val.npy
X_test_seen.npy
X_test_zero_day.npy
y_train.npy
y_val.npy
y_test_seen.npy
y_test_zero_day.npy
family_train.npy
family_val.npy
family_test_seen.npy
family_test_zero_day.npy
row_id_train.npy
row_id_val.npy
row_id_test_seen.npy
row_id_test_zero_day.npy
feature_schema.json
```

Nếu thiếu một file, preprocess chạy lại nếu stage được phép.

### 7.2 `_features_ready(feature_set)`

Yêu cầu đủ:

```text
F_train.npy
F_val.npy
F_test_seen.npy
F_test_zero_day.npy
memae_feature_schema.json
```

### 7.3 Compatibility checks

Pipeline không chỉ kiểm tra file tồn tại. Nó còn kiểm tra:

```text
MemAE checkpoint input_dim == data/processed/{experiment}/X_train.npy.shape[1]
Feature schema D_value == processed input dim
Feature schema include_raw_input == CLI --include-raw-input-features
Feature schema raw_input_feature_patterns == CLI --raw-input-feature-pattern
```

Nếu preprocess vừa chạy hoặc checkpoint/feature schema không tương thích, downstream sẽ tự chạy lại theo thứ tự MemAE → feature export → XGBoost → fusion nếu stage được phép. Nếu user bắt đầu quá muộn, ví dụ `--start-at features` nhưng MemAE checkpoint đã lệch input dim, script raise lỗi rõ và yêu cầu chạy từ `memae` hoặc sớm hơn.

### 7.4 Fusion model check

Fusion readiness yêu cầu:

```text
artifacts/fusion/{fusion_artifact}/fusion_model.joblib
```

Nếu feature hoặc XGBoost vừa chạy lại, fusion cũng bị coi là stale và train lại.

---

## 8. Clean data option

CLI:

```bash
--clean-data
```

Khi bật, script xóa:

```text
data/splits
data/processed
data/features
```

Nó không xóa:

```text
data/interim
artifacts
reports
```

Tùy chọn này hữu ích để giải phóng dữ liệu trung gian nhưng cần cẩn thận: nếu artifacts model còn đó mà data/features bị xóa, stages readiness sẽ quyết định chạy lại theo điều kiện hiện có.

---

## 9. Fingerprints trong summary

Summary JSON ghi fingerprint cho:

```text
clean_data
window_config
memae_config
xgboost_config
```

`_file_fingerprint()` ghi:

```text
path
size_bytes
mtime_ns
```

Với config files, `hash_file=True` nên có thêm:

```text
sha256
```

Điều này giúp xác định report được tạo từ đúng config nào. Clean parquet hiện chỉ fingerprint bằng size/mtime, không hash toàn file để tránh chi phí lớn.

---

## 10. Directory map

### 10.1 Raw and interim data

```text
data/cicids2017/
  *.csv
  *.parquet

data/interim/
  cicids2017_clean.parquet
  column_schema.json
  attack_family_mapping.json
  data_quality_report.json
```

`data/interim` là output của cleaning và là input chung cho split/preprocess.

### 10.2 Splits

```text
data/splits/{experiment}/
  train.csv
  val.csv
  test_seen.csv
  test_zero_day.csv
  split_manifest.json
```

CSV split chỉ chứa:

```text
row_id
attack_family
original_label
binary_label
```

Nó không chứa feature values.

### 10.3 Processed arrays

```text
data/processed/{experiment}/
  X_*.npy
  y_*.npy
  family_*.npy
  row_id_*.npy
  feature_schema.json
```

`X_*.npy` là input cho MemAE. `feature_schema.json` là contract feature order sau preprocessing.

### 10.4 MemAE features

```text
data/features/{feature_set}/
  F_*.npy
  memae_feature_schema.json
```

`F_*.npy` là input cho XGBoost và score fusion.

### 10.5 Artifacts

```text
artifacts/preprocessors/{experiment}/preprocessor.joblib
artifacts/memae/{feature_set}/memae_best.pt
artifacts/memae/{feature_set}/training_log.json
artifacts/xgboost/{feature_set}/xgboost_model.json
artifacts/xgboost/{feature_set}/threshold.json
artifacts/xgboost/{feature_set}/feature_importance.json
artifacts/xgboost/{feature_set}/training_log.json
artifacts/fusion/{fusion_artifact}/fusion_model.joblib
artifacts/fusion/{fusion_artifact}/training_log.json
artifacts/fusion/{fusion_artifact}/val_score.npy
```

### 10.6 Reports

```text
reports/runs/{timestamp}_{suffix}/
  full_pipeline_all_families_summary.json       # chỉ có khi chạy đủ DEFAULT_FAMILIES
  full_pipeline_all_families_summary.md         # chỉ có khi chạy đủ DEFAULT_FAMILIES
  full_pipeline_{suffix}_summary.json
  full_pipeline_{suffix}_summary.md
  zero_day_{family}_.../
    detector_calibration_report_*.json
    detector_calibration_report_*.md
    fusion_calibration_report_*.json
    fusion_calibration_report_*.md
```

Mỗi lần chạy pipeline tạo một thư mục riêng dưới `reports/runs/`, vì vậy các report cũ được giữ nguyên và không còn khái niệm ghi đè stable/latest.

---

## 11. Chạy từng đoạn pipeline

### 11.1 Chỉ tạo split

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families botnet \
  --start-at split \
  --stop-after split \
  --force-retrain
```

Output chính:

```text
data/splits/zero_day_botnet_host_disjoint_zdr5/
```

### 11.2 Chạy tới processed features

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families botnet \
  --start-at preprocess \
  --stop-after preprocess
```

Nếu split chưa có, lệnh này không tạo split vì `start-at preprocess` bỏ qua stage split. Cần đảm bảo split đã tồn tại.

### 11.3 Train lại model từ MemAE trở đi

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families botnet \
  --start-at memae \
  --stop-after reports \
  --force-retrain
```

Yêu cầu `data/processed/{experiment}` đã tồn tại.

### 11.4 Chỉ sinh lại reports

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families botnet \
  --start-at reports \
  --stop-after reports
```

Yêu cầu đầy đủ artifacts model/features.

---

## 12. Tests

File test hiện có:

```text
tests/test_pipeline_hardening.py
```

Các nhóm kiểm tra:

1. Window features nâng cao có tên đúng và finite.
2. Primary candidate selection tôn trọng FPR cap.
3. Summary suffix cho single-family không ghi nhầm tên generic.
4. MemAE export có thể append raw processed input và chọn raw feature theo pattern.

Chạy:

```bash
.venv/bin/python -m unittest tests/test_pipeline_hardening.py
```

Đây không phải full integration test toàn dataset, nhưng bảo vệ những contract dễ vỡ nhất:

- Window feature naming/value.
- Candidate selection.
- Report naming.
- MemAE feature dimension/schema.

---

## 13. Checklist vận hành benchmark

1. Xác nhận raw data có trong `data/cicids2017`.
2. Nếu thay cleaning/split/preprocess/window config, chạy lại từ stage tương ứng với `--force-retrain`.
3. Nếu thay config nhưng giữ suffix cũ, artifact cũ có thể bị reuse; nên đổi suffix hoặc force retrain.
4. Sau preprocess, kiểm tra `feature_schema.json`.
5. Sau MemAE export, kiểm tra `memae_feature_schema.json`.
6. Sau XGBoost, kiểm tra `training_log.json` và `best_iteration`.
7. Sau reports, đọc primary summary và từng calibration report cho family có FPR drift cao.
8. Không so sánh hai run nếu config fingerprint/suffix khác mà chưa ghi chú rõ.
