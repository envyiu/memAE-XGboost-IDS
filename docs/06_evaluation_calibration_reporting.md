# Module: Evaluation, Calibration & Reporting

File này đi sâu vào module báo cáo:

```text
src/evaluation/detector_calibration.py
src/evaluation/fusion_calibration.py
scripts/run_full_pipeline_all_families.py::_summarize_family()
```

Mục tiêu của module này là đánh giá zero-day theo FPR budget, không dùng threshold train-time một cách mù quáng.

---

## 1. Triết lý đánh giá

Benchmark zero-day của dự án dùng nguyên tắc:

```text
Chọn threshold bằng benign calibration scores
Đánh giá threshold đó nguyên vẹn trên test_zero_day
Không tune threshold bằng zero-day attack
```

Metric trung tâm:

```text
Z-DR = tỷ lệ attack rows trong test_zero_day được dự đoán malicious
FPR  = false positives / benign rows
```

`test_zero_day` chứa:

- Benign rows để đo FPR.
- Attack rows thuộc đúng family bị giấu để đo Z-DR.

Các seen attacks không được giữ trong `test_zero_day`; nếu group zero-day kéo theo seen attack rows, split module sẽ exclude chúng khỏi CSV `test_zero_day.csv`.

---

## 2. FPR budgets

Default trong cả detector và fusion calibration:

```python
DEFAULT_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
```

Full pipeline truyền từ CLI:

```bash
--fpr-budgets 0.001,0.005,0.01,0.02,0.05
--max-observed-test-fpr 0.05
```

Hai khái niệm khác nhau:

- `target_fpr`: ngân sách dùng để chọn threshold trên calibration benign.
- `max_observed_test_fpr`: cap để đánh dấu PASS/FAIL trên `test_zero_day`.

Một row có thể chọn threshold theo target 1% nhưng observed test FPR vượt 5%, khi đó status FAIL.

---

## 3. Calibration modes

Hai mode:

```text
val_only
val_plus_test_seen_benign
```

Default recipe chính:

```text
val_plus_test_seen_benign
```

Trong `detector_calibration.py`:

```python
val_benign = val_score[val["family"] == "benign"]
if calibration_mode == "val_only":
    return val_benign
if calibration_mode == "val_plus_test_seen_benign":
    seen_benign = test_seen_score[test_seen["family"] == "benign"]
    return concatenate([val_benign, seen_benign])
```

Mode default dùng thêm benign từ `test_seen`, nhưng không dùng seen attack và không dùng zero-day. Mục đích là có phân phối benign calibration rộng hơn để threshold FPR ổn định hơn.

---

## 4. Threshold selection

### 4.1 Detector calibration

`detector_calibration._threshold_for_fpr(benign_score, target_fpr)`:

1. Nếu số unique score <= 10, thêm jitter rất nhỏ `uniform(0, 1e-7)` để tránh tie quá nhiều.
2. Sort score.
3. Lấy các threshold unique.
4. Tính FPR nếu predict malicious khi `score >= threshold`.
5. Chọn threshold đầu tiên có `fpr <= target_fpr`.
6. Nếu không có threshold hợp lệ, dùng percentile fallback.

Pseudocode:

```text
thresholds = sorted(unique(score))
fpr(threshold) = count(score >= threshold) / n
valid = thresholds where fpr <= target_fpr
choose smallest threshold in valid
```

Chọn smallest threshold hợp lệ nghĩa là recall sẽ cao nhất trong các threshold thỏa FPR budget.

### 4.2 Fusion calibration

`fusion_calibration._threshold_for_fpr()` tương tự nhưng không thêm jitter. Nếu không có threshold đạt target, nó dùng:

```text
nextafter(max(benign_score), +inf)
```

để tạo threshold có calibration FPR bằng 0.

---

## 5. Metrics

Detector dùng `_metrics_from_pred()`:

```text
tn, fp, fn, tp
precision
recall
z_dr
f1
fpr
```

Trong đó:

```text
attack_mask = family != "benign"
z_dr = mean(pred[attack_mask] == 1)
fpr = fp / (fp + tn)
```

Với `test_zero_day`, attack_mask chính là zero-day family. Với `val` hoặc `test_seen`, attack_mask là seen attacks.

### 5.1 Recall và Z-DR

Trong binary setup, nếu split chỉ có benign và một loại attack, recall và Z-DR thường giống nhau. Tuy nhiên code vẫn ghi cả hai vì:

- `recall` đến từ `sklearn` trên `y_true`.
- `z_dr` được tính trực tiếp bằng `family != "benign"`.

Z-DR rõ nghĩa hơn trong báo cáo zero-day.

### 5.2 FPR drift ratio

```text
fpr_drift_ratio = observed_test_fpr / calibration_fpr
```

Denominator được chặn tối thiểu `1e-12` để tránh chia 0.

Nếu drift ratio lớn, threshold đạt calibration FPR nhưng không giữ được FPR trên test benign. Đây là tín hiệu distribution shift hoặc calibration benign chưa đại diện.

---

## 6. Detector calibration report

Entry:

```python
generate_detector_calibration_report(
  experiment,
  feature_set,
  xgboost_artifact,
  calibration_mode,
  fpr_budgets,
  max_observed_test_fpr,
  report_dir,
)
```

Load:

```text
data/processed/{experiment}/y_*.npy
data/processed/{experiment}/family_*.npy
data/features/{feature_set}/F_*.npy
artifacts/xgboost/{xgboost_artifact}/xgboost_model.json
artifacts/xgboost/{xgboost_artifact}/threshold.json
```

Score:

```text
xgb_score   = XGBoost predict_proba(F)[:, 1]
memae_score = F[:, 0]
```

Report sinh bốn nhóm chính.

### 6.1 `xgboost_fixed_fpr`

Với mỗi target FPR:

```text
threshold = threshold_for_fpr(xgb_calibration_benign, target_fpr)
evaluate xgb_score trên val, test_seen, test_zero_day
```

Mỗi row có:

```text
model_name = xgboost
score_key = xgb_score
selection_rule
threshold
calibration_fpr
validation_fpr
target_fpr
validation
test_seen
test_zero_day
observed_test_fpr
fpr_drift_ratio
fpr_cap
fpr_status
```

### 6.2 `memae_fixed_fpr`

Tương tự XGBoost nhưng dùng:

```text
score_key = memae_reconstruction_error
score = F[:, 0]
```

MemAE score càng cao càng bất thường.

### 6.3 `or_fusion_budget_grid`

OR fusion không train model mới. Rule:

```text
malicious if xgboost_score >= tx OR memae_score >= tm
```

Với mỗi total budget, code chia budget theo 11 ratio:

```text
memae_budget = total_budget * ratio
xgb_budget   = total_budget * (1 - ratio)
ratio in linspace(0, 1, 11)
```

Threshold `tm`, `tx` được chọn riêng trên calibration benign của từng score. `calibration_fpr` của row được ghi xấp xỉ bằng tổng hai calibration FPR:

```text
tm.calibration_fpr + tx.calibration_fpr
```

Observed FPR vẫn được đo trực tiếp bằng OR prediction trên split.

### 6.4 `or_fusion_fine_grid_top_by_validation_recall`

Fine grid dùng 21 điểm chia budget cho mỗi total budget. Sau đó lọc:

```text
validation.fpr <= max(fpr_budgets)
```

và lấy top 10 theo validation recall.

Nhóm này dùng để phân tích thêm, không nhất thiết là primary result.

---

## 7. Fusion calibration report

Entry:

```python
generate_fusion_calibration_report(
  experiment,
  feature_set,
  xgboost_artifact,
  fusion_artifact,
  calibration_mode,
  fpr_budgets,
  max_observed_test_fpr,
  report_dir,
)
```

Load:

```text
artifacts/xgboost/{xgboost_artifact}/xgboost_model.json
artifacts/fusion/{fusion_artifact}/fusion_model.joblib
data/features/{feature_set}/F_val.npy
data/features/{feature_set}/F_test_seen.npy
data/features/{feature_set}/F_test_zero_day.npy
```

Với mỗi split:

```text
xgb_score = predict_proba(XGBoost, F)
memae_score = F[:, 0]
fusion_score = logistic.predict_proba(_fusion_features(xgb_score, memae_score))[:, 1]
```

Sau đó chọn threshold theo FPR budget trên fusion benign calibration scores và đánh giá trên val/test_seen/test_zero_day.

Report JSON gồm:

```text
experiment
feature_set
xgboost_artifact
fusion_artifact
benchmark_mode
processed_feature_count
memae_input_dim
threshold_fit_scope
calibration_mode
fpr_budgets
max_observed_test_fpr
rows
candidate_rows
```

`rows` và `candidate_rows` hiện giống nhau để full pipeline gom candidate dễ hơn.

---

## 8. Markdown reports

Detector report markdown:

```text
reports/.../detector_calibration_report_{xgboost_artifact}.md
```

Các bảng:

- XGBoost Fixed FPR
- MemAE Fixed FPR
- OR Fusion Budget Grid
- OR Fusion Fine Grid: Top By Validation Recall

Fusion report markdown:

```text
reports/.../fusion_calibration_report_{fusion_artifact}.md
```

Bảng gồm:

```text
model
target_fpr
threshold
cal_fpr
val_fpr
test_fpr
fpr_drift_ratio
zdr
f1
status
```

Markdown phục vụ đọc nhanh; JSON là artifact đầy đủ cho phân tích tự động.

---

## 9. Full pipeline summary

Trong `scripts/run_full_pipeline_all_families.py`, `_summarize_family()`:

1. Gọi detector calibration.
2. Gọi fusion calibration.
3. Load hai JSON report.
4. Lấy row 1% FPR cho XGBoost và logistic fusion để ghi `xgboost_1pct`, `fusion_1pct`.
5. Gom toàn bộ candidate.
6. Chọn primary candidate.

Output mỗi family:

```text
experiment
feature_set
fusion_artifact
report_paths
xgboost_1pct
fusion_1pct
primary_result
candidate_results
support
family
```

Sau tất cả family, summary ghi:

```text
reports/metrics/full_pipeline_all_families_summary.json
reports/metrics/full_pipeline_all_families_summary.md
reports/run_{timestamp}/full_pipeline_{suffix}_summary.json
reports/run_{timestamp}/full_pipeline_{suffix}_summary.md
```

Generic all-family summary chỉ ghi khi `families == DEFAULT_FAMILIES`. Nếu chạy một family, chỉ ghi summary trong thư mục run timestamp.

---

## 10. Primary candidate selection

Candidate đến từ:

```text
detector.candidate_rows
fusion.candidate_rows
```

Nếu report cũ không có `candidate_rows`, code fallback sang:

```text
xgboost_fixed_fpr
memae_fixed_fpr
or_fusion_budget_grid
fusion.rows
```

### 10.1 Lọc theo FPR cap

```python
passing = [
  row for row in rows
  if row["test_zero_day"]["fpr"] <= max_observed_test_fpr
]
```

Nếu có passing, chọn max theo `_candidate_selection_key()`.

Nếu không có passing, chọn row có `test_zero_day.fpr` thấp nhất và đánh dấu `primary_selection_status = FAIL`.

### 10.2 Selection key

Key:

```text
1. zd_zdr
2. cal_quality
3. target_fpr
4. model_priority
5. -test_fpr
```

Trong đó:

```text
cal_quality = 1 - min(abs(cal_fpr - target_fpr) / target_fpr, 1)
model_priority:
  xgboost = 3
  memae = 3
  or_fusion = 2
  logistic_fusion = 1
```

Ưu tiên đầu tiên là Z-DR thực tế trên zero-day miễn FPR cap pass. Nếu Z-DR bằng nhau, threshold bám target FPR tốt hơn được ưu tiên.

Lưu ý: comment trong Markdown nói "maximizes seen-validation recall" nhưng implementation hiện tại ưu tiên `test_zero_day.z_dr`. Khi diễn giải kết quả, nên tin implementation trong `_candidate_selection_key()`.

---

## 11. Support

`support` trong family summary lấy từ row XGBoost 1%:

```python
support = tp + fn trên test_zero_day
```

Vì support là số attack zero-day trong split, nó không phụ thuộc model nếu split giống nhau.

Aggregate summary tính:

```text
macro_zdr    = mean(zdr across families)
weighted_zdr = average(zdr, weights=support)
worst_zdr    = min(zdr across families)
```

Macro phản ánh công bằng theo family. Weighted bị family lớn chi phối.

---

## 12. Cách đọc report đúng

Khi xem một family:

1. Mở `full_pipeline_*_summary.md` để xem primary result.
2. Nếu primary FAIL, mở detector/fusion report để xem row nào vượt FPR.
3. So sánh `cal_fpr`, `val_fpr`, `test_fpr`.
4. Nếu `cal_fpr` thấp nhưng `test_fpr` cao, nghi drift benign.
5. Nếu `zdr` thấp ở mọi model dù FPR pass, feature/model chưa bắt được family đó.
6. Nếu MemAE tốt hơn XGBoost, zero-day family có thể khác seen attack nhưng vẫn anomalous.
7. Nếu XGBoost tốt hơn MemAE, feature representation supervised đang generalize qua family.

---

## 13. Checklist khi sửa evaluation

1. Không chọn threshold dựa trên `test_zero_day` attack.
2. Nếu thêm model mới, đưa row vào `candidate_rows` với cùng schema.
3. Luôn ghi `calibration_mode` và `fpr_budgets` vào JSON.
4. Giữ `test_zero_day.fpr`, `test_zero_day.z_dr`, `fpr_status` cho summary.
5. Nếu thay selection key, cập nhật mô tả trong Markdown summary để không lệch implementation.
6. Khi thêm metric, ghi vào JSON trước; Markdown chỉ là view rút gọn.
