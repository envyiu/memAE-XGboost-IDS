# memAE-XGboost-IDS

IDS2 là pipeline benchmark zero-day intrusion detection trên CIC-IDS2017. Recipe chính hiện tại là `zdr5` với split `host_disjoint_window`: mỗi lượt giữ lại một attack family làm zero-day, train trên benign + các seen attack family còn lại, rồi đánh giá detection rate của family bị giấu dưới giới hạn FPR.

## Benchmark Chính

- Families chính: `web_attack`, `botnet`, `portscan`, `ddos`, `dos`, `brute_force`.
- Families bị loại khỏi benchmark chính: `heartbleed` và `infiltration` vì support dưới 100 mẫu.
- Split chính: host-disjoint theo `source_file + source_ip`.
- Window features: `configs/window_features_zdr5.yaml`.
- MemAE: `configs/memae_targeted.yaml`.
- TabTransformer numeric extractor: `configs/tabtrans_zdr5.yaml`.
- XGBoost: `configs/xgboost_zdr5.yaml`.
- Calibration: `val_plus_test_seen_benign`.
- FPR budgets: `0.001,0.005,0.01,0.02,0.05`.
- Gate: observed `test_zero_day` FPR phải `<= 0.05`.

## Pipeline

```text
raw CIC-IDS2017
  -> clean parquet + column schema
  -> host-disjoint LOFO split
  -> train-fit + model_selection_val holdout carved from train groups
  -> preprocessing + window/context features
  -> MemAE train trên benign train
  -> export MemAE-derived F_* features + raw processed window/context features
  -> train XGBoost trên F_*
  -> train logistic score fusion artifact
  -> OR fusion calibration reports
  -> per-run summary trong reports/runs/
```

Preprocessing hiện có một điểm quan trọng: continuous numeric features vẫn bị quantile-clip theo train split, nhưng các context/indicator features như `ctx_*`, `is_*`, `*_is_*`, `*_indicator*` không bị clip. Điều này giữ được tín hiệu binary hiếm, ví dụ port `8080` của botnet.

Split hiện tạo thêm `model_selection_val` từ train groups. MemAE/XGBoost/fusion/report sẽ ưu tiên split này cho model selection, early stopping, threshold/validation diagnostics nếu file tồn tại; `val` host-disjoint cũ vẫn được giữ như split diagnostic, không còn là điểm nghẽn khi nó có quá ít attack family.

Benchmark chính đã cố định là `or_fusion`: XGBoost và MemAE vẫn được train/export vì là hai nguồn score nội bộ, nhưng report chính không còn phát hành benchmark standalone cho hai model này.

Nhánh thử nghiệm TabTransformer có thể dùng lại toàn bộ `data/interim`, `data/splits` và `data/processed` hiện có. Nó chỉ thay tầng representation sau preprocessing: train `src/models/tabtrans`, export latent `F_*`, train XGBoost, rồi report XGBoost-only. Nhánh này không dùng score fusion vì không có MemAE reconstruction score.

## Chạy Local

Tạo môi trường:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy toàn bộ benchmark:

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families all \
  --force-retrain
```

Chạy riêng một family:

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families botnet \
  --force-retrain
```

Chạy lại từ một stage cụ thể:

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families botnet \
  --start-at preprocess \
  --stop-after reports \
  --force-retrain
```

Chỉ sinh lại report nếu artifacts đã tồn tại:

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families botnet \
  --start-at reports \
  --stop-after reports
```

Chạy nhánh TabTransformer từ dữ liệu processed sẵn có:

```bash
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --architecture tabtrans \
  --families botnet \
  --start-at memae \
  --stop-after reports \
  --force-retrain
```

Với `--architecture tabtrans`, nếu không truyền `--variant-suffix`, runner dùng suffix `tabtrans_zdr5` để không ghi đè feature set/artifact MemAE cũ.

Mỗi lần chạy report tạo một thư mục riêng:

```text
reports/runs/{timestamp}_{summary_suffix}/
```

## CLI Quan Trọng

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--families` | `all` | Chọn `all` hoặc danh sách family cụ thể. |
| `--architecture` | `memae` | Chọn representation backend: `memae` hoặc `tabtrans`. |
| `--start-at` | `split` | Stage bắt đầu: `split`, `preprocess`, `memae`, `features`, `xgboost`, `fusion`, `reports`. |
| `--stop-after` | `reports` | Stage cuối cùng cần chạy. |
| `--force-retrain` | off | Bỏ qua cache/artifact hiện có và chạy lại các stage được phép. |
| `--clean-data` | off | Xóa `data/splits`, `data/processed`, `data/features` trước khi chạy. |
| `--report-root` | `reports/runs` | Root chứa thư mục report cho từng run. |
| `--model-selection-ratio` | `0.15` | Tỉ lệ group trong train được giữ lại làm `model_selection_val` cho model selection/validation. |
| `--include-raw-input-features` | on | Append raw processed `X_*` vào representation feature `F_*`. |
| `--no-raw-input-features` | off | Tắt append raw processed input để chạy representation-only. |
| `--raw-input-feature-pattern` | unset | Khi append raw input, chỉ chọn feature name chứa pattern này. Có thể truyền nhiều lần. |
| `--preprocess-device` | `cpu` | Backend transform preprocessing: `cpu`, `cuda`, `auto`. |
| `--preprocess-num-workers` | `0` | Số process theo `source_file` cho CPU `full_source_file`; `0` là auto. |
| `--max-observed-test-fpr` | `0.05` | FPR cap để đánh dấu PASS/FAIL trong summary. |

## Chạy Trên Kaggle

Nếu đã upload nguyên thư mục `data/` lên Kaggle Dataset, runner sẽ tự tìm thư mục có đủ `data/interim`, `data/splits`, `data/processed`, symlink các thư mục này vào `/kaggle/working/memAE-XGboost-IDS/data`, rồi mặc định chạy từ stage `memae`. `data/features` vẫn nằm ở `/kaggle/working` để có thể ghi feature mới.

```bash
cd /kaggle/working/memAE-XGboost-IDS
python -u scripts/run_kaggle_pipeline.py \
  --architecture tabtrans \
  --families all \
  --force-retrain
```

Test nhanh một family:

```bash
python -u scripts/run_kaggle_pipeline.py \
  --architecture tabtrans \
  --families botnet \
  --force-retrain
```

Nếu auto-detect không đúng dataset, truyền rõ:

```bash
python -u scripts/run_kaggle_pipeline.py \
  --prepared-data-dir /kaggle/input/<dataset-slug>/data \
  --architecture tabtrans \
  --families botnet \
  --force-retrain
```

Runner Kaggle mặc định chạy `--architecture tabtrans`, dùng `configs/tabtrans_kaggle_t4x2.yaml` và `configs/xgboost_kaggle_gpu.yaml`; nếu truyền `--architecture memae` thì dùng `configs/memae_kaggle_t4x2.yaml`. Config TabTransformer dùng micro-batch `1024` + gradient accumulation `8` vì 2 T4 không gộp VRAM thành một GPU 30GB. Với prepared data, `--clean-data` chỉ xóa output sinh lại (`data/features`, artifacts, reports), không xóa `splits/processed` đã symlink từ Kaggle input.

## Cấu Trúc Thư Mục

```text
configs/                  Recipe chính và biến thể Kaggle GPU
scripts/                  Entrypoint local/Kaggle và setup environment
src/data/                 Clean CIC-IDS2017 và split LOFO
src/preprocessing/        IDSPreprocessor và preprocess orchestration
src/features/window/      Context/window feature engineering
src/features/             Export representation-derived features
src/models/memae/         MemAE model + training
src/models/tabtrans/      Numeric TabTransformer representation model + training
src/models/xgboost/       XGBoost feature-set detector
src/models/fusion/        Logistic score fusion
src/evaluation/           Detector/fusion calibration reports
tests/                    Smoke/unit tests cho hardening
data/interim/             Clean parquet và schema
data/splits/              Split CSV theo experiment
data/processed/           X_*, y_*, family_*, row_id_* arrays
data/features/            F_* representation feature arrays
artifacts/                Preprocessors và model artifacts
reports/runs/             Report riêng cho từng lần chạy
```

## Đọc Kết Quả

Mở file summary trong thư mục run mới nhất:

```text
reports/runs/{timestamp}_{suffix}/full_pipeline_{suffix}_summary.md
```

Các cột chính:

- `selected_model`: `or_fusion` với MemAE, `xgboost` với TabTransformer.
- `observed_test_fpr`: FPR thực tế trên benign trong `test_zero_day`.
- `zdr`: recall trên zero-day attack family.
- `f1`: F1 trên `test_zero_day`.
- `status`: `PASS` nếu observed FPR không vượt cap.

Nên đọc thêm detector calibration report bên trong thư mục family nếu primary nhìn bất thường. Nhánh MemAE phát hành benchmark `or_fusion`; nhánh TabTransformer phát hành report XGBoost-only.

## Tests

```bash
.venv/bin/python -m py_compile $(find scripts src tests -name '*.py')
.venv/bin/python -m unittest tests.test_pipeline_hardening
```

Các test quan trọng hiện có:

- Primary benchmark cố định là `or_fusion`.
- Report directory phải là per-run và không ghi đè.
- MemAE checkpoint input dim phải khớp processed feature dim.
- Context/indicator features không bị quantile-clip.
- MemAE export có thể append raw processed input.
- TabTransformer train/export tạo schema representation riêng và không dùng fusion.
