# memAE-XGboost-IDS
# (IDS2 Zero-Day Benchmark)

Pipeline chính hiện tại là `zdr5` trên benchmark `host_disjoint_window`.

## Recipe Chính

- Families: `web_attack`, `botnet`, `portscan`, `ddos`, `dos`, `brute_force`
- Loại khỏi benchmark chính: `heartbleed`, `infiltration` vì support dưới 100
- Split: host-disjoint theo `source_file + source_ip`
- Window config: `configs/window_features_zdr5.yaml`
- MemAE config: `configs/memae_targeted.yaml`
- XGBoost config: `configs/xgboost_zdr5.yaml`
- Calibration: `val_plus_test_seen_benign`
- FPR budgets: `0.001,0.005,0.01,0.02,0.05`
- Gate: observed `test_zero_day` FPR `<= 0.05`

## Chạy Pipeline Chính

Chạy toàn bộ family:

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py --families all --force-retrain
```

Chạy riêng một family:

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py --families botnet --force-retrain
```

## Chạy Trên Kaggle

Nếu repo nằm ở `/kaggle/working/memAE-XGboost-IDS` và dataset nằm ở `/kaggle/input/datasets/envyiu/cicids2017`, dùng notebook `IDS2_Kaggle.ipynb` hoặc chạy:

```bash
cd /kaggle/working/memAE-XGboost-IDS
python -u scripts/run_kaggle_pipeline.py --families all --clean-data
```

Để test nhanh một family:

```bash
python -u scripts/run_kaggle_pipeline.py --families botnet --clean-data
```

Runner Kaggle mặc định dùng `--preprocess-device cuda` và ghi temp preprocess vào `/kaggle/working/ids2_preprocess_tmp`.
Với máy 30GB RAM + 2xT4, runner dùng config `configs/memae_kaggle_t4x2.yaml` để bật MemAE `DataParallel`, AMP, batch train `16384`, batch eval/export lớn hơn; XGBoost dùng `configs/xgboost_kaggle_gpu.yaml` với `device: cuda`.

Audit artifact chính:

```bash
.venv/bin/python scripts/audit_pipeline_artifacts.py --families all
```

Diagnostics cho một family:

```bash
.venv/bin/python scripts/generate_family_diagnostics.py \
  --experiment zero_day_botnet_host_disjoint_zdr5 \
  --feature-set zero_day_botnet_host_disjoint_zdr5_targetsel_zdr5 \
  --target-fpr 0.05 \
  --top-k 50
```

## Cấu Trúc

- `configs/`: chỉ giữ config của recipe chính.
- `scripts/`: entrypoint chạy pipeline, train từng model, report, audit, diagnostics.
- `src/data/`: clean data và split zero-day.
- `src/preprocessing/`: tạo `X_*` processed arrays.
- `src/features/`: window features và export MemAE-derived `F_*`.
- `src/models/`: MemAE, XGBoost feature-set detector, score fusion.
- `src/evaluation/`: calibration report cho detector/fusion.
- `data/interim/`: clean parquet và schema.
- `data/splits/`: split CSV của recipe chính.
- `data/processed/`: `X_*`, labels, family, row ids của recipe chính.
- `data/features/`: `F_*` MemAE features của recipe chính.
- `artifacts/`: model/preprocessor artifacts của recipe chính.
- `reports/metrics/`: summary, audit, calibration, diagnostics.
