# Module: Data & Preprocessing

Module này bao gồm toàn bộ quy trình từ raw CSV đến ma trận numpy đã chuẩn hóa, sẵn sàng đưa vào mô hình. Gồm 4 file chính:

```
src/data/clean_cicids2017.py       → Đọc, chuẩn hóa, map nhãn
src/data/split_zero_day.py         → Chia LOFO split
src/preprocessing/preprocessor.py  → IDSPreprocessor class
src/preprocessing/run_preprocessing.py → Orchestrate toàn bộ
```

---

## 1. `clean_cicids2017.py` — Làm sạch dữ liệu thô

### 1.1 Chức năng

File này đọc tất cả CSV/Parquet trong `data/cicids2017/`, chuẩn hóa tên cột, map nhãn tấn công sang attack family, xử lý dữ liệu bẩn, và ghi ra `data/interim/cicids2017_clean.parquet`.

### 1.2 Chi tiết từng bước trong `clean_dataset()`

**Bước 1: Đọc raw tables** (`load_raw_tables()`)

```python
# Quét tất cả CSV hoặc Parquet trong data_dir
paths = sorted(data_dir.glob("*.csv"))  # ưu tiên CSV, fallback Parquet
```

Mỗi file được:
- Normalize tên cột: `" Flow Duration "` → `flow_duration` (loại ký tự đặc biệt, lowercase, gộp underscore)
- Deduplicate cột trùng tên: `flow_id`, `flow_id_1` (CIC-IDS2017 có cột trùng)
- Canonicalize alias: `fwd_packets_length_total` → `total_length_of_fwd_packets` (do khác phiên bản CIC-IDS2017)
- Parse timestamp: thử `dayfirst=True` trước, fallback `dayfirst=False`
- Thêm `source_file` (tên file gốc) và `day_of_week` (parse từ tên file)

**Bước 2: Assign row_id và map nhãn**

```python
df.insert(0, "row_id", np.arange(len(df)))  # ID duy nhất toàn dataset
df["original_label"] = df["label"].map(normalize_label)
df["attack_family"] = df["original_label"].map(ATTACK_FAMILY_MAPPING)
```

Bảng ánh xạ `ATTACK_FAMILY_MAPPING`:

| Nhãn gốc (original_label) | Họ tấn công (attack_family) |
|---|---|
| `BENIGN`, `Benign` | `benign` |
| `FTP-Patator`, `SSH-Patator` | `brute_force` |
| `DoS slowloris`, `DoS Hulk`, `DoS GoldenEye`, `DoS Slowhttptest` | `dos` |
| `Heartbleed` | `heartbleed` |
| `Web Attack - Brute Force`, `XSS`, `Sql Injection` | `web_attack` |
| `Infiltration` | `infiltration` |
| `Bot` | `botnet` |
| `PortScan` | `portscan` |
| `DDoS` | `ddos` |

Hàm `normalize_label()` xử lý Unicode bẩn (`\ufffd`), chuẩn hóa dấu gạch ngang, khoảng trắng.

**Bước 3: Xử lý giá trị bất thường**

```python
# Thay thế Inf → NaN trong tất cả cột số
for col in numeric_cols:
    mask = np.isinf(df[col])
    df.loc[mask, col] = np.nan

# Loại bỏ hàng duplicate (trừ row_id)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"])
```

**Bước 4: Phân loại cột**

```python
protected = {"row_id", "label", "original_label", "attack_family", "source_file", "day_of_week"}
leakage_candidates = {"flow_id", "source_ip", "destination_ip", "source_port", "destination_port", "timestamp"}
# leakage_candidates bị đánh dấu dropped_columns nhưng VẪN GIỮ trong parquet
# (vì window features cần chúng)
constant_columns = [c for c in candidate_features if df[c].nunique() <= 1]
```

> **Quan trọng:** Các cột leakage KHÔNG bị xóa khỏi parquet — chúng chỉ bị đánh dấu trong `column_schema.json` là `dropped_columns`. Module preprocessing sau sẽ tự chọn cột nào dùng.

**Bước 5: Output**

| File output | Nội dung |
|---|---|
| `cicids2017_clean.parquet` | DataFrame đầy đủ (tất cả cột) |
| `column_schema.json` | Metadata: danh sách cột numerical, categorical, dropped |
| `attack_family_mapping.json` | Bảng ánh xạ nhãn |
| `data_quality_report.json` | Thống kê: số hàng gốc, trùng lặp, NaN, Inf |

---

## 2. `split_zero_day.py` — Chia dữ liệu LOFO

### 2.1 Triết lý thiết kế

Không chia theo **row** mà chia theo **group** để tránh data leakage. Mỗi group là một tổ hợp `(source_file, source_ip)` (mode `host`) hoặc `(source_file, source_ip, destination_ip, destination_port)` (mode `exact_flow`).

**Lý do:** Nếu chia theo row, cùng một IP source có thể xuất hiện ở cả train và test → mô hình "nhớ" pattern IP thay vì học hành vi thực sự.

### 2.2 Thuật toán `create_leave_one_family_out_split()`

**Input:**
- `zero_day_family`: Họ tấn công cần giấu (e.g., `"dos"`)
- `group_columns`: Cột dùng để nhóm (mặc định: `source_file, source_ip, destination_ip, destination_port`)
- Tỷ lệ chia: `train=0.70, val=0.10, test_seen=0.10, test_zero_day_benign=0.10`

**Bước 1: Xây group key**

```python
# Hash tổ hợp cột thành uint64 key duy nhất
group_key = hash_pandas_object(df[["source_file", "source_ip", ...]])
```

**Bước 2: Tách group zero-day**

```python
zero_day_groups = {g for g in groups if any row in g has attack_family == zero_day_family}
split_groups["test_zero_day"] = zero_day_groups
```

Tất cả group chứa ít nhất 1 flow của họ zero-day bị gán vào `test_zero_day`. Điều này đảm bảo **không có leak giữa train và test**.

**Bước 3: Chia seen attacks theo family**

Với mỗi họ tấn công **không phải** benign và **không phải** zero-day:
```python
for family in seen_families:
    groups = family_groups[family]
    allocate(groups, train=0.7, val=0.1, test_seen=0.1)
```

Thuật toán `_weighted_group_split()`:
1. Shuffle groups (seed-controlled)
2. Sắp xếp giảm dần theo kích thước group
3. Với mỗi group, gán vào split có **remaining quota lớn nhất** (greedy bin-packing)

Edge cases:
- Nếu family chỉ có 1 group → gán toàn bộ vào train
- Nếu family có 2 groups mà train rỗng → gán group lớn vào train, nhỏ vào val

**Bước 4: Chia benign**

Groups benign (chưa bị gán bởi seen attacks) được chia thành 4 phần: train, val, test_seen, test_zero_day (để test_zero_day có benign cho FPR measurement).

**Bước 5: Validation & output**

```python
# ASSERT: không có row_id trùng giữa các split
# ASSERT: không có group trùng giữa các split
# ASSERT: zero_day_family KHÔNG xuất hiện trong train hoặc val
```

Output: 4 file CSV + `split_manifest.json` chứa thống kê chi tiết.

### 2.3 Cấu trúc split

| Split | Benign | Seen attacks | Zero-day attack |
|---|---|---|---|
| `train` | ✅ 70% | ✅ 70% | ❌ |
| `val` | ✅ 10% | ✅ 10% | ❌ |
| `test_seen` | ✅ 10% | ✅ 10% | ❌ |
| `test_zero_day` | ✅ 10% | ❌ (filtered out) | ✅ 100% |

---

## 3. `preprocessor.py` — IDSPreprocessor

### 3.1 Pipeline nội bộ

```python
class IDSPreprocessor:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
```

### 3.2 Chi tiết `fit()` và `transform()`

**Sanitize (trước cả fit lẫn transform):**
```python
def _sanitize(self, data):
    X = data.astype(np.float32)
    # Các cột "invalid negative" (e.g., fwd_header_length) → NaN nếu < 0
    for idx in self.invalid_negative_indices:
        X[X[:, idx] < 0, idx] = np.nan
    return X
```

**Fit (chỉ trên train split):**
```python
def fit(self, data):
    X = self._sanitize(data)
    self.lower_bounds_ = np.nanquantile(X, 0.001, axis=0)  # clip bottom 0.1%
    self.upper_bounds_ = np.nanquantile(X, 0.999, axis=0)  # clip top 0.1%
    X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
    self.pipeline.fit(X)  # fit median imputer + standard scaler
```

**Transform:**
```python
def transform(self, data, device="cpu", batch_rows=262_144):
    X = self._sanitize(data)
    X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
    X = self.pipeline.transform(X)  # impute NaN → median, then standardize
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # final safety
    return X.astype("float32")
```

`device="auto"` hoặc `device="cuda"` dùng Torch/CUDA cho phần transform theo batch. Logic fit vẫn giữ artifact scikit-learn như cũ để median, clipping bounds và scaler tương thích với pipeline hiện tại. Trên Colab có thể bật bằng:

```bash
python scripts/run_full_pipeline_all_families.py \
  --preprocess-device auto \
  --preprocess-batch-rows 262144 \
  --preprocess-tmp-dir /content/ids2_preprocess_tmp
```

Trên Colab/Google Drive nên dùng `--preprocess-tmp-dir` trỏ tới `/content/...` để tránh ghi `open_memmap` trực tiếp lên Drive. Nếu process chết với exit code `-7`, nguyên nhân thường là `SIGBUS` từ mmap/FUSE hoặc thiếu quota đĩa, không phải Python exception.

### 3.3 Serialization

```python
# Save: joblib.dump(self, path)
# Load: joblib.load(path) → IDSPreprocessor
```

---

## 4. `run_preprocessing.py` — Orchestration

### 4.1 `preprocess_experiment()`

Đây là hàm orchestration lớn nhất của preprocessing, kết nối tất cả thành phần.

**Luồng chính:**

```
1. Đọc column_schema.json → xác định feature columns
2. Resolve window config → xác định cột phụ thuộc cần load
3. Đọc parquet THEO TỪNG SOURCE_FILE (tránh load toàn bộ ~3M rows vào RAM)
4. Tính window features nếu enabled
5. Fit IDSPreprocessor trên train split (có sampling nếu quá lớn)
6. Transform tất cả split → ghi ra numpy memmap
```

### 4.2 Window scope

Tham số `window_scope` quyết định cách tính window features:

| Mode | Mô tả |
|---|---|
| `full_source_file` | Tính window trên TOÀN BỘ source file, sau đó mới filter split. **Sát thực tế nhất.** |
| `split_only` | Chỉ tính window trên rows thuộc split hiện tại. |
| `causal_past_only` | Val: tính trên train+val. Test: tính trên train+val+test. |

Mặc định: `full_source_file` — mô phỏng hệ thống IDS real-time nhìn thấy mọi flow trong file gốc.

### 4.3 Memory efficiency: Streaming per source file

```python
for source_file in files:
    chunk = pd.read_parquet(clean_path, filters=[("source_file", "==", source_file)])
    if window_enabled:
        chunk, _ = add_window_features(chunk, window_cfg)
    # extract only rows belonging to current split
    part = chunk.loc[chunk_ids]
    transformed = preprocessor.transform(part[features])
    X_memmap[offset:offset+len(part)] = transformed
```

Kỹ thuật này cho phép xử lý dataset lớn mà không cần load toàn bộ vào RAM.

### 4.4 Output structure

```
data/processed/{experiment}/
├── X_train.npy         # (N_train, D) float32 memmap
├── X_val.npy           # (N_val, D)
├── X_test_seen.npy     # (N_test_seen, D)
├── X_test_zero_day.npy # (N_test_zd, D)
├── y_train.npy         # (N_train,) int64, binary: 0=benign, 1=attack
├── y_val.npy
├── y_test_seen.npy
├── y_test_zero_day.npy
├── family_train.npy    # (N_train,) object, e.g. "benign", "brute_force"
├── family_val.npy
├── family_test_seen.npy
├── family_test_zero_day.npy
├── row_id_train.npy    # (N_train,) int64
├── row_id_val.npy
├── row_id_test_seen.npy
├── row_id_test_zero_day.npy
└── feature_schema.json # Metadata: feature_order, scaler_type, window config
```

### 4.5 Preprocessor fit sampling

Khi train split quá lớn (> `fit_sample_rows`, mặc định 400K):

```python
sample_ratio = min(1.0, fit_sample_rows / train_total)
# Random sample mỗi chunk với tỷ lệ này
# → tổng ~400K rows dùng fit preprocessor
```

Đây là **stratified random** trên mỗi source file chunk, đảm bảo mọi source đều đóng góp.
