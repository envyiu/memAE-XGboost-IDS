# Kiến Trúc Hệ Thống Phát Hiện Xâm Nhập Zero-Day (IDS2)

## 1. Tổng quan đề tài

**IDS2** là hệ thống phát hiện xâm nhập mạng (Intrusion Detection System) được thiết kế chuyên biệt cho bài toán **phát hiện tấn công zero-day** — tức các dạng tấn công **chưa từng xuất hiện trong dữ liệu huấn luyện**.

### 1.1 Bài toán

Hệ thống IDS truyền thống dựa trên chữ ký (signature-based) chỉ phát hiện được các tấn công đã biết. Khi xuất hiện một dạng tấn công mới (zero-day), hệ thống hoàn toàn "mù". IDS2 giải quyết vấn đề này bằng cách kết hợp hai chiến lược:

1. **Memory-Augmented Autoencoder (MemAE)** — Học mô hình phân phối lưu lượng bình thường (benign), phát hiện bất thường qua sai số tái tạo.
2. **XGBoost** — Bộ phân loại gradient boosting mạnh, sử dụng cả đặc trưng gốc lẫn đặc trưng trích xuất từ MemAE.
3. **Score Fusion** — Kết hợp điểm từ cả hai mô hình để tối ưu hiệu quả phát hiện.

### 1.2 Phương pháp đánh giá: Leave-One-Family-Out (LOFO)

Để mô phỏng đúng kịch bản zero-day, hệ thống sử dụng giao thức đánh giá **LOFO**:

- Với mỗi họ tấn công (ví dụ: DoS, Botnet, PortScan...), **loại bỏ hoàn toàn** họ đó khỏi tập huấn luyện và validation.
- Đánh giá khả năng phát hiện họ tấn công "chưa biết" đó trên tập test riêng biệt (`test_zero_day`).
- Lặp lại cho tất cả 6 họ tấn công chính, tổng hợp kết quả thành benchmark.

### 1.3 Dataset

Hệ thống sử dụng **CIC-IDS2017** — bộ dữ liệu tiêu chuẩn quốc tế cho nghiên cứu IDS, gồm:

| Họ tấn công | Mô tả | Số lượng xấp xỉ |
|---|---|---|
| `brute_force` | FTP-Patator, SSH-Patator | ~13,800 |
| `dos` | Slowloris, Slowhttptest, Hulk, GoldenEye | ~252,600 |
| `web_attack` | Brute Force, XSS, SQL Injection | ~2,180 |
| `infiltration` | Infiltration (loại do ít mẫu) | ~36 |
| `botnet` | Bot | ~1,966 |
| `portscan` | PortScan | ~158,900 |
| `ddos` | DDoS | ~128,000 |
| `heartbleed` | Heartbleed (loại do ít mẫu) | ~11 |
| `benign` | Lưu lượng bình thường | ~2,273,000 |

> **Lưu ý:** `heartbleed` (11 mẫu) và `infiltration` (36 mẫu) bị loại khỏi benchmark chính do quá ít mẫu, chỉ dùng cho diagnostics.

---

## 2. Kiến trúc mô hình

### 2.1 Pipeline tổng quan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        run_full_pipeline_all_families.py                    │
│                          (Entry Point duy nhất)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌──────────┐    ┌────────┐    ┌──────────┐    ┌─────────┐  │
│  │  SPLIT   │───▶│PREPROCESS│───▶│ MEMAE  │───▶│ FEATURES │───▶│ XGBOOST │  │
│  │          │    │          │    │ TRAIN  │    │ EXPORT   │    │  TRAIN  │  │
│  └─────────┘    └──────────┘    └────────┘    └──────────┘    └─────────┘  │
│                                                                     │       │
│                                                               ┌─────▼─────┐ │
│                                                               │  FUSION   │ │
│                                                               │  TRAIN    │ │
│                                                               └─────┬─────┘ │
│                                                                     │       │
│                                                               ┌─────▼─────┐ │
│                                                               │  REPORTS  │ │
│                                                               │(Calibrate)│ │
│                                                               └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Các giai đoạn (Stages)

Pipeline gồm 7 giai đoạn tuần tự, mỗi giai đoạn có thể skip nếu artifact đã tồn tại:

| # | Stage | Module | Mô tả |
|---|---|---|---|
| 1 | `split` | `src.data.split_zero_day` | Tạo split LOFO cho mỗi họ tấn công |
| 2 | `preprocess` | `src.preprocessing.run_preprocessing` | Chuẩn hóa dữ liệu + tính window features |
| 3 | `memae` | `src.models.memae.train_memae` | Huấn luyện MemAE trên benign-only |
| 4 | `features` | `src.features.export_memae_features` | Trích xuất feature vector từ MemAE |
| 5 | `xgboost` | `src.models.xgboost.train_feature_set` | Huấn luyện XGBoost phân loại |
| 6 | `fusion` | `src.models.fusion.train_score_fusion` | Huấn luyện Logistic Regression fusion |
| 7 | `reports` | `src.evaluation.*` | Calibrate threshold + sinh báo cáo |

### 2.3 Luồng dữ liệu chi tiết

```
data/cicids2017/*.csv
        │
        ▼
  clean_dataset()  ──▶  data/interim/cicids2017_clean.parquet
        │                       + column_schema.json
        ▼                       + attack_family_mapping.json
  create_leave_one_family_out_split()
        │
        ▼
  data/splits/zero_day_{family}/
        ├── train.csv          (row_id, attack_family, binary_label)
        ├── val.csv
        ├── test_seen.csv
        ├── test_zero_day.csv
        └── split_manifest.json
        │
        ▼
  preprocess_experiment()
        │
        ▼
  data/processed/zero_day_{family}_{suffix}/
        ├── X_{split}.npy      (float32 feature matrix, memmap)
        ├── y_{split}.npy      (binary labels)
        ├── family_{split}.npy (attack family strings)
        ├── row_id_{split}.npy
        └── feature_schema.json
        │
        ▼
  train_memae()  ──▶  artifacts/memae/{feature_set}/memae_best.pt
        │
        ▼
  export_features()
        │
        ▼
  data/features/{feature_set}/
        ├── F_{split}.npy      (MemAE-derived feature vectors)
        └── memae_feature_schema.json
        │
        ├──▶ train_xgboost()  ──▶  artifacts/xgboost/{feature_set}/
        │                            ├── xgboost_model.json
        │                            ├── threshold.json
        │                            └── feature_importance.json
        │
        └──▶ train_score_fusion()  ──▶  artifacts/fusion/{artifact}/
                                          ├── fusion_model.joblib
                                          └── val_score.npy
```

---

## 3. Cấu trúc thư mục dự án

```
IDS2/
├── configs/                          # Cấu hình YAML cho các thành phần
│   ├── memae_targeted.yaml           # Siêu tham số MemAE
│   ├── xgboost_zdr5.yaml            # Siêu tham số XGBoost
│   └── window_features_zdr5.yaml    # Cấu hình window feature engineering
│
├── scripts/
│   └── run_full_pipeline_all_families.py   # Entry point duy nhất
│
├── src/
│   ├── data/                         # Thu thập & chia dữ liệu
│   │   ├── clean_cicids2017.py       # Đọc, chuẩn hóa cột, map nhãn, loại bỏ trùng lặp
│   │   └── split_zero_day.py         # Chia LOFO split theo group-level
│   │
│   ├── preprocessing/                # Tiền xử lý pipeline
│   │   ├── preprocessor.py           # IDSPreprocessor (impute, clip, scale)
│   │   └── run_preprocessing.py      # Orchestrate preprocessing + window features
│   │
│   ├── features/                     # Feature engineering
│   │   ├── window/                   # Window features package
│   │   │   ├── config.py             # DEFAULT_WINDOW_CONFIG, merge, resolve
│   │   │   ├── names.py              # window_feature_names()
│   │   │   ├── rolling.py            # rolling_sum, entropy, sliding counters
│   │   │   ├── time_window.py        # Time-based sliding window stats
│   │   │   └── engine.py             # add_window_features() orchestration
│   │   └── export_memae_features.py  # Trích xuất feature từ MemAE checkpoint
│   │
│   ├── models/
│   │   ├── memae/
│   │   │   ├── model.py              # Kiến trúc MemAE (encoder, memory, decoder)
│   │   │   └── train_memae.py        # Training loop + model selection
│   │   ├── xgboost/
│   │   │   ├── train_feature_set.py  # XGBoost training + threshold optimization
│   │   │   └── threshold_optimizer.py # F1-based threshold search
│   │   └── fusion/
│   │       └── train_score_fusion.py  # Logistic Regression score fusion
│   │
│   ├── evaluation/
│   │   ├── detector_calibration.py   # XGBoost + MemAE + OR-fusion calibration
│   │   └── fusion_calibration.py     # Logistic fusion calibration
│   │
│   └── utils/
│       ├── io.py                     # read/write JSON, YAML, ensure_dir
│       └── seed.py                   # Global random seed management
│
├── tests/
│   └── test_pipeline_hardening.py    # 4 integration tests
│
├── data/                             # (Gitignored) Dữ liệu trung gian
│   ├── cicids2017/                   # Raw CSV/Parquet
│   ├── interim/                      # Cleaned parquet + schema
│   ├── splits/                       # LOFO split CSVs
│   ├── processed/                    # Preprocessed numpy arrays
│   └── features/                     # MemAE-derived feature matrices
│
├── artifacts/                        # (Gitignored) Model checkpoints
│   ├── preprocessors/
│   ├── memae/
│   ├── xgboost/
│   └── fusion/
│
└── reports/                          # Kết quả đánh giá
    └── runs/
        └── {timestamp}_{suffix}/     # Mỗi lần chạy tạo 1 thư mục riêng
```

---

## 4. Metric đánh giá chính

### 4.1 Zero-Day Detection Rate (Z-DR)

Metric cốt lõi là **Z-DR** — tỷ lệ phát hiện đúng của các mẫu tấn công zero-day (họ tấn công bị giấu hoàn toàn khỏi training):

$$Z\text{-}DR = \frac{\text{TP}_{\text{zero-day}}}{\text{TP}_{\text{zero-day}} + \text{FN}_{\text{zero-day}}}$$

### 4.2 FPR Budget Constraint

Mọi threshold đều được chọn dưới ràng buộc **FPR ≤ budget** (thường 1%):

- Threshold được calibrate trên benign scores từ `val + test_seen_benign`.
- Sau đó đánh giá trực tiếp trên `test_zero_day` mà **không điều chỉnh lại threshold**.
- Nếu observed FPR trên test vượt cap (5%), kết quả đánh dấu `FAIL`.

### 4.3 Candidate Selection

Pipeline tự động chọn **primary candidate** từ tất cả model/threshold combinations:
1. Lọc candidate có `test_zero_day.fpr ≤ max_observed_test_fpr`
2. Chọn candidate theo tín hiệu `test_seen` rồi `validation`, không dùng zero-day recall để tune
3. Nếu không candidate nào pass → chọn candidate có FPR thấp nhất (đánh dấu `FAIL`)

Điểm này cố ý bảo thủ: `test_zero_day` chỉ được dùng để kiểm tra FPR cap và báo cáo kết quả cuối, không dùng attack label zero-day để chọn model/threshold.

---

## 5. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Mô hình anomaly detection | PyTorch (MemAE) |
| Mô hình phân loại | XGBoost |
| Fusion | scikit-learn (Logistic Regression) |
| Tiền xử lý | scikit-learn (StandardScaler, SimpleImputer) |
| Feature engineering | NumPy (hand-crafted rolling windows) |
| Dữ liệu | Pandas, Parquet, NumPy memmap |
| Serialization | joblib, torch.save, JSON |
| Reproducibility | Global seed management (numpy, torch, random) |

---

## 6. Cách chạy

### 6.1 Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Chạy full pipeline

```bash
# Đặt raw CIC-IDS2017 CSV/Parquet vào data/cicids2017/
# Chạy toàn bộ pipeline cho tất cả 6 họ tấn công:
.venv/bin/python scripts/run_full_pipeline_all_families.py --families all
```

### 6.3 Các tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--families` | `all` | Chọn họ tấn công: `all`, `dos`, `botnet`, ... |
| `--start-at` | `split` | Bắt đầu từ stage nào |
| `--stop-after` | `reports` | Dừng sau stage nào |
| `--force-retrain` | `false` | Bỏ qua cache, train lại từ đầu |
| `--clean-data` | `false` | Xóa sạch `data/splits`, `data/processed`, `data/features` |
| `--include-raw-input-features` | `false` | Ghép raw preprocessed features vào MemAE features |
| `--calibration-mode` | `val_plus_test_seen_benign` | Nguồn calibration benign |
| `--fpr-budgets` | `0.001,0.005,0.01,0.02,0.05` | Các ngưỡng FPR cần đánh giá |

### 6.4 Chạy tests

```bash
.venv/bin/python -m unittest tests.test_pipeline_hardening -v
```
