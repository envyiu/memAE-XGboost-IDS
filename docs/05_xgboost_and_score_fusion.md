# Module: XGBoost & Score Fusion

File này đi sâu vào hai tầng supervised sau MemAE:

```text
src/models/xgboost/train_feature_set.py
src/models/xgboost/threshold_optimizer.py
src/models/fusion/train_score_fusion.py
```

XGBoost là detector chính trên feature set do MemAE export. Score fusion là tầng logistic regression kết hợp score XGBoost và score MemAE.

---

## 1. Vị trí trong pipeline

Luồng:

```text
data/features/{feature_set}/F_*.npy
data/processed/{experiment}/y_*.npy + family_*.npy
  -> train_xgboost_feature_set()
  -> artifacts/xgboost/{feature_set}/xgboost_model.json
  -> train_score_fusion()
  -> artifacts/fusion/{feature_set}_{fusion_suffix}/fusion_model.joblib
```

XGBoost train trước fusion. Fusion cần model XGBoost đã train để tạo `xgb_score`.

---

## 2. XGBoost input

`train_xgboost_feature_set()` load:

```text
F_train.npy
F_val.npy
y_train.npy
y_val.npy
family_train.npy
```

Trong đó:

- `F_*` là MemAE-derived features, có thể kèm raw processed input.
- `y_*` là nhãn binary: benign=0, attack=1.
- `family_train.npy` dùng cho family-balanced sampling nếu bật.

XGBoost không trực tiếp đọc `X_*.npy`. Nó train trên `F_*.npy`.

---

## 3. Sampling và cân bằng family

Hàm `_sample_xy()` có hai mode.

### 3.1 Sampling thường

Nếu không bật `family_balance`:

```text
Nếu max_samples rỗng hoặc len(X) <= max_samples:
  dùng toàn bộ
Ngược lại:
  random choice max_samples không hoàn lại
```

### 3.2 Family-balanced sampling

Config chính:

```yaml
training:
  max_train_samples: 600000
  max_val_samples: 150000
  family_balance: true
  max_samples_per_attack_family: 80000
  benign_to_attack_ratio: 2.5
```

Khi bật `family_balance` và có `family_train`:

1. Duyệt từng family trong train.
2. Bỏ qua `benign` ở bước attack.
3. Với mỗi attack family, lấy tối đa `max_samples_per_attack_family`.
4. Tính tổng attack được lấy: `atk_total`.
5. Lấy benign tối đa:

```text
target_benign = atk_total * benign_to_attack_ratio
```

6. Nếu tổng sample vẫn vượt `max_train_samples`, random tiếp xuống `max_train_samples`.
7. Sort index để đọc memmap ổn định hơn.

Mục tiêu là tránh việc benign hoặc một family lớn như DoS/DDoS áp đảo các family nhỏ trong supervised training.

### 3.3 Validation sampling

Validation dùng `_sample_xy(F_val, y_val, max_val_samples, seed+1)` nhưng không truyền family balance. Nghĩa là val sample là random theo row, không cân bằng family.

---

## 4. Class imbalance trong XGBoost

Sau sampling:

```python
neg = count(y_train_sample == 0)
pos = count(y_train_sample == 1)
scale_pos_weight = neg / pos if pos else 1.0
```

`scale_pos_weight` được inject vào params của `xgb.XGBClassifier`.

Điều này xử lý mất cân bằng benign/attack ở cấp binary label. Family balance xử lý mất cân bằng giữa các attack family. Hai cơ chế này bổ sung nhau.

---

## 5. XGBoost config chính

`configs/xgboost_zdr5.yaml`:

```yaml
binary_detection:
  objective: binary:logistic
  eval_metric: aucpr
  tree_method: hist
  n_estimators: 1500
  early_stopping_rounds: 100
  learning_rate: 0.03
  max_depth: 8
  min_child_weight: 2
  subsample: 0.85
  colsample_bytree: 0.85
  gamma: 0.1
  reg_alpha: 0.0005
  reg_lambda: 1.0
  seed: 42
```

Ý nghĩa thực tế:

- `binary:logistic`: output là xác suất attack.
- `aucpr`: phù hợp dữ liệu mất cân bằng hơn accuracy.
- `hist`: train nhanh hơn trên dataset lớn.
- `early_stopping_rounds`: dừng nếu val AUC-PR không cải thiện.
- `max_depth=6`: hạn chế overfit vào seen families tốt hơn cấu hình sâu hơn.
- `gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`: tăng regularization để giảm fit noise.
- `subsample` và `colsample_bytree`: regularization theo row/column.
- `n_estimators=500`: đủ dư địa so với các best_iteration lịch sử nhưng tránh kéo dài fit khi internal eval đã gần bão hòa.

Code lấy `early_stopping_rounds` ra rồi đưa lại vào params vì API XGBoost hiện tại nhận được qua constructor.

Nếu official validation split có quá ít attack samples (`training.min_eval_attack_samples`), training tạo một stratified holdout từ train sample cho early stopping và permutation importance. Official `val` vẫn được giữ cho calibration/reporting; holdout nội bộ chỉ dùng để tránh model selection dựa trên 0-3 positive samples.

---

## 6. Prediction contract

`src.utils.scoring.predict_prob(model, X)` xử lý best iteration:

```python
best_iteration = getattr(model, "best_iteration", None)
if best_iteration is not None:
    return model.predict_proba(X, iteration_range=(0, best_iteration + 1))[:, 1]
return model.predict_proba(X)[:, 1]
```

Điều này đảm bảo inference dùng đúng số cây tốt nhất khi early stopping hoạt động. Utility này cũng áp `selected_feature_indices` khi artifact XGBoost được retrain bằng feature selection.

Nếu viết code mới để evaluate XGBoost, không nên gọi trực tiếp `model.predict_proba(X)` mà bỏ qua `best_iteration`, vì có thể dùng cả các cây sau điểm tốt nhất.

---

## 7. Threshold optimizer của XGBoost

`threshold_optimizer.py` rất đơn giản:

```python
for threshold in np.linspace(min_t, max_t, steps):
  y_pred = y_prob >= threshold
  score = f1_score(y_true, y_pred)
  chọn threshold có F1 cao nhất
```

Config:

```yaml
threshold:
  metric: f1
  min: 0.01
  max: 0.99
  steps: 99
```

Lưu ý quan trọng: threshold này là threshold F1 trên validation sample và được ghi vào:

```text
artifacts/xgboost/{feature_set}/threshold.json
```

Nhưng benchmark chính không dựa hoàn toàn vào threshold F1 này. Các report calibration sẽ chọn threshold theo FPR budget từ benign calibration scores. `threshold.json` chủ yếu là artifact train-time và baseline tham chiếu.

---

## 8. XGBoost artifacts

Sau training:

```text
artifacts/xgboost/{feature_set}/
  xgboost_model.json
  threshold.json
  feature_importance.json
  feature_selection.json
  training_log.json
```

### 8.1 `training_log.json`

Gồm:

```text
experiment
feature_set
feature_dims
model_feature_dims
feature_schema
train_samples_used
fit_samples_used
val_samples_used
eval_samples_used
eval_source
official_val_positive_samples
eval_positive_samples
class_counts_train
class_counts_fit
family_counts_train_used
family_balance_enabled
scale_pos_weight
best_iteration
best_score
threshold
feature_selection
evals_result
```

Đây là nơi kiểm tra:

- XGBoost đang train trên bao nhiêu chiều.
- Model cuối cùng dùng bao nhiêu chiều sau feature selection.
- `eval_source` là official val hay train stratified holdout.
- Family balance có thật sự bật không.
- Số sample từng family có hợp lý không.
- `best_iteration` có nhỏ hơn `n_estimators` không.
- AUC-PR validation tiến triển ra sao.

### 8.2 `feature_selection.json`

Khi `training.feature_selection=true`, training fit model ban đầu, tính permutation importance trên eval split hiệu lực, bỏ các feature có importance `<= feature_selection_threshold`, rồi retrain model cuối trên feature đã chọn. Recipe `zdr5` chính hiện tắt feature selection để giữ kết quả ổn định và tránh bỏ nhầm tín hiệu zero-day; logic này chỉ là tùy chọn thử nghiệm. Các MemAE scalar nền như `re_scalar`, `attn_entropy`, `attn_sparsity`, `attn_max` được giữ mặc định nếu feature selection được bật. Metadata lưu:

```text
selected_indices
selected_feature_count
original_feature_count
protected_indices
importances_mean
importances_std
```

Downstream calibration và score fusion đọc metadata này để apply đúng subset trước khi gọi XGBoost.

### 8.3 `feature_importance.json`

Gồm ba kiểu importance từ booster:

```text
gain
weight
cover
```

Do feature trong `F_*` không có tên chi tiết từng cột, key thường là dạng `f0`, `f1`, ... Cần map ngược bằng `memae_feature_schema.json` nếu muốn diễn giải sâu.

---

## 9. Score fusion

Entry:

```python
train_score_fusion(experiment, feature_set, xgboost_artifact, fusion_artifact)
```

Fusion train logistic regression trên score-level features, không dùng toàn bộ `F_*`.

### 9.1 Input

```text
artifacts/xgboost/{xgboost_artifact}/xgboost_model.json
data/features/{feature_set}/F_train.npy
data/features/{feature_set}/F_val.npy
data/processed/{experiment}/y_train.npy
data/processed/{experiment}/y_val.npy
```

### 9.2 Score nguồn

```python
xgb_train = predict_prob_batched(model, F_train)
xgb_val = predict_prob_batched(model, F_val)
memae_train = F_train[:, 0]
memae_val = F_val[:, 0]
```

`src.utils.scoring.predict_prob_batched()` dự đoán theo batch 100,000 rows để tránh load/predict một mảng quá lớn cùng lúc.

MemAE score là `F[:, 0]`, tức reconstruction error scalar.

---

## 10. Fusion feature formula

`_fusion_features(xgb_score, memae_score)` tạo 8 chiều:

```text
1. xgb_score
2. xgb_logit = log(xgb_score / (1 - xgb_score))
3. memae_score
4. log1p_memae_score = log(1 + max(memae_score, 0))
5. xgb_times_log1p_memae
6. max_xgb_tanh_memae = max(xgb_score, tanh(log1p_memae_score / 10))
7. abs_xgb_memae_diff
8. min_xgb_tanh_memae = min(xgb_score, tanh(log1p_memae_score / 10))
```

Chi tiết:

- `xgb_score` giữ xác suất thô từ supervised detector.
- `xgb_logit` mở rộng vùng gần 0/1 để logistic dễ học tuyến tính hơn.
- `memae_score` giữ reconstruction error thô.
- `log1p_memae_score` giảm skew của reconstruction error.
- Tích `xgb * log_memae` biểu diễn đồng thuận giữa hai detector.
- `max(xgb, tanh(...))` giống một feature OR mềm.
- `abs(xgb - memae)` biểu diễn mức bất đồng giữa supervised detector và anomaly score.
- `min(xgb, tanh(...))` biểu diễn đồng thuận bảo thủ.

Code clip XGBoost score vào `[1e-6, 1 - 1e-6]` để tránh logit vô cực.

---

## 11. Calibrated logistic fusion

Model:

```python
base = LogisticRegression(
  class_weight="balanced",
  max_iter=2000,
  solver="lbfgs",
  random_state=42,
)
clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
```

Nó train trên:

```text
X_train = _fusion_features(xgb_train, memae_train)
y_train
```

Sau đó ghi:

```text
artifacts/fusion/{fusion_artifact}/fusion_model.joblib
artifacts/fusion/{fusion_artifact}/training_log.json
artifacts/fusion/{fusion_artifact}/val_score.npy
```

`val_score.npy` là fusion probability trên val, dùng để kiểm tra nhanh nhưng `fusion_calibration.py` vẫn có thể tự tính lại score từ model.

---

## 12. Fusion artifact naming

Trong full pipeline:

```text
feature_set     = {experiment}_{variant_suffix}
fusion_artifact = {feature_set}_{fusion_suffix}
```

Với default:

```text
zero_day_dos_host_disjoint_zdr5_targetsel_zdr5_scorefusion
```

Điều này tách rõ:

- XGBoost artifact theo feature set.
- Fusion artifact theo feature set + fusion recipe.

Nếu thử nhiều kiểu fusion trên cùng XGBoost, chỉ cần đổi `fusion_suffix`.

---

## 13. Quan hệ giữa XGBoost, MemAE và fusion trong benchmark

Report calibration không chỉ dùng logistic fusion. Nó sinh candidate từ:

```text
xgboost_fixed_fpr
memae_fixed_fpr
or_fusion_budget_grid
logistic_fusion
```

Vì vậy XGBoost và MemAE vẫn được đánh giá độc lập. Logistic fusion là một candidate bổ sung, không thay thế các detector gốc.

Pipeline summary chọn primary candidate từ tất cả candidate theo ràng buộc FPR cap, nhưng selection không tune theo zero-day recall. Do đó một family có thể chọn XGBoost, family khác chọn MemAE hoặc fusion; khi `val/test_seen` thiếu attack signal, cần đọc thêm candidate table vì primary không nhất thiết là candidate có F1 cao nhất trên `test_zero_day`.

---

## 14. Rủi ro và điểm cần chú ý

### 14.1 XGBoost học shortcut từ MemAE feature

Nếu `F_*` append raw processed input toàn bộ, XGBoost có thể dựa nhiều vào raw feature hơn MemAE representation. Điều này không sai, nhưng khi báo cáo cần ghi rõ `include_raw_input_features`.

### 14.2 Threshold F1 không đồng nhất với FPR budget

`threshold.json` tối ưu F1, còn benchmark chọn threshold theo FPR. Không nên dùng `threshold.json` để kết luận Z-DR dưới FPR cap.

### 14.3 Validation sample bị giới hạn

`max_val_samples=150000` có thể làm early stopping và threshold F1 dựa trên sample thay vì full val. Calibration report sau đó dùng toàn bộ `F_val`/`F_test_seen` khi load từ file, nên số liệu final vẫn đáng tin hơn train-time threshold.

### 14.4 Fusion train trên train, calibrate trên val/test_seen benign

Calibrated logistic fusion học từ train labels và calibrate xác suất nội bộ bằng 3-fold isotonic calibration. Threshold của fusion được chọn sau đó bằng benign calibration scores. Không được tune threshold trên `test_zero_day`.

### 14.5 Contract cột `F[:, 0]`

Fusion và detector calibration đều giả định `F[:, 0]` là MemAE reconstruction error. Nếu export feature thay đổi thứ tự, fusion hỏng ngầm.

---

## 15. Checklist khi chỉnh model supervised

1. Nếu thay feature export, train lại XGBoost.
2. Nếu thay XGBoost params, ghi rõ config mới và không dùng artifact cũ.
3. Nếu thay `_predict_prob`, đảm bảo vẫn dùng `best_iteration`.
4. Nếu đổi fusion feature formula, cập nhật `fusion_feature_names` trong training log.
5. Nếu thêm model fusion mới, thêm candidate vào reporting thay vì ghi đè semantic của logistic fusion.
6. Sau training, kiểm tra `training_log.json` của XGBoost và fusion trước khi đọc summary.
