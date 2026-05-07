# IDS2 Option B Implementation Plan — Code Refactor + Generalization

## Goal

Giảm code dư thừa trong codebase và cải thiện khả năng khái quát hóa (generalization) của model, đặc biệt là giải quyết thảm họa botnet Z-DR=0.334.

---

## Botnet Root Cause Analysis

> [!CAUTION]
> **Root cause đã xác định:** Botnet failure là lỗi hệ thống, không phải thiếu features.

### Bằng chứng cụ thể:

**1. XGBoost hoàn toàn thất bại: `best_iteration = 0`**
- AUC-PR = 0.000082 (ngẫu nhiên thuần)
- XGBoost chỉ có **1 cây** (0 iteration = cây gốc), output score = 0.485-0.515 cho MỌI sample
- Chỉ có **7 giá trị unique** trong toàn bộ botnet predictions
- Tương đương: model không học được GÌ từ features

**2. MemAE reconstruction error overlap nghiêm trọng nhưng KHÔNG hoàn toàn vô vọng:**
```
Botnet recon: mean=68.99, p90=134.88, p99=208.72
Benign recon: mean=32.07, p90=64.55, p99=256.33
```
- Ở benign p90 threshold: 38% botnet detected → **có tín hiệu nhưng cần FPR quá cao**
- Ở benign p99 threshold: chỉ 0.2% detected → vì benign long-tail (max=3394) che lấp botnet

**3. MemAE val_seen_attack chỉ có 3 samples!**
- Model selection metric `seen_recall_at_benign_fpr` không có nghĩa thống kê với 3 mẫu
- `best_selection_value = 0.0` → MemAE được chọn bằng val_loss thay vì attack detection

**4. So sánh với families khác (best_iteration):**

| Family | XGB best_iter | AUC-PR | MemAE val_attack_samples |
|---|---|---|---|
| **botnet** | **0** | **0.000082** | **3** |
| brute_force | 5 | 0.1345 | 229 |
| portscan | 1 | 0.1910 | 229 |
| dos | 82 | 0.2999 | 229 |
| ddos | 117 | 0.3520 | 229 |
| web_attack | 256 | 0.3223 | 229 |

> [!IMPORTANT]
> **Kết luận:** Botnet split có vấn đề nghiêm trọng — XGBoost không learn được (best_iter=0), MemAE chỉ có 3 validation attack samples. Vấn đề nằm ở cách features được trích xuất từ MemAE (242 dims vs 332 input dims → info loss) VÀ model quá yếu ở giai đoạn supervised khi seen attacks KHÔNG đủ tương tự botnet để supervised signal transfer.

---

## Proposed Changes

### Phase 1: Code Redundancy Refactor (Ưu tiên: Code Quality)

---

#### [NEW] [scoring.py](file:///home/envyiu/Music/IDS2/src/utils/scoring.py)

Tập trung **tất cả** hàm scoring/prediction dùng chung, xóa bỏ 4 bản copy `_predict_prob()`:

```python
# Hàm hợp nhất từ 4 file:
def predict_prob(model, X)                     # từ detector_cal, fusion_cal, train_feature_set, train_score_fusion
def predict_prob_batched(model, X, batch_size)  # từ train_score_fusion  
def threshold_for_fpr(benign_score, target_fpr, add_jitter=False)  # hợp nhất 2 bản
def metrics_from_pred(y_true, family, pred)     # từ detector_cal + fusion_cal
def metrics_at_threshold(y_true, family, score, threshold)
def fpr_drift_ratio(test_fpr, cal_fpr)
def calibration_benign_scores(mode, val, test_seen, val_score, test_seen_score)
DEFAULT_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
```

**Xử lý khác biệt nhỏ giữa 2 bản `_threshold_for_fpr`:**
- `detector_calibration.py`: thêm jitter khi unique scores ≤ 10, fallback = percentile
- `fusion_calibration.py`: không jitter, fallback = `nextafter(max, inf)`
- **Giải pháp:** tham số `add_jitter=False`, `fallback_mode='percentile'|'nextafter'`

---

#### [NEW] [reporting.py](file:///home/envyiu/Music/IDS2/src/utils/reporting.py)

Hợp nhất markdown/rendering utilities:

```python
def markdown_table(rows, columns)               # từ cả 2 calibration files
def render_calibration_rows(rows)               # từ cả 2 calibration files
def with_fpr_status(row, max_observed_test_fpr) # từ detector_cal._with_status
```

---

#### [MODIFY] [detector_calibration.py](file:///home/envyiu/Music/IDS2/src/evaluation/detector_calibration.py)

- Xóa: `_predict_prob()`, `_threshold_for_fpr()`, `_metrics_from_pred()`, `_metrics_at_threshold()`, `_fpr_drift_ratio()`, `_with_status()`, `_calibration_benign_scores()`, `_markdown_table()`, `_render_rows()`, `DEFAULT_FPR_BUDGETS`
- Import từ `src.utils.scoring` và `src.utils.reporting`
- Giữ nguyên: `_make_score_row()`, `_load_split()`, `_write_markdown()`, `generate_detector_calibration_report()` (chỉ sửa import calls)

---

#### [MODIFY] [fusion_calibration.py](file:///home/envyiu/Music/IDS2/src/evaluation/fusion_calibration.py)

- Xóa: `_predict_prob()`, `_threshold_for_fpr()`, `_metrics()`, `_fpr_drift_ratio()`, `_calibration_scores()`, `_markdown_table()`, `_render_rows()`, `DEFAULT_FPR_BUDGETS`
- Import từ `src.utils.scoring` và `src.utils.reporting`

---

#### [MODIFY] [train_feature_set.py](file:///home/envyiu/Music/IDS2/src/models/xgboost/train_feature_set.py)

- Xóa: `_predict_prob()`
- Import: `from src.utils.scoring import predict_prob`

---

#### [MODIFY] [train_score_fusion.py](file:///home/envyiu/Music/IDS2/src/models/fusion/train_score_fusion.py)

- Xóa: `_predict_prob()`, `_predict_prob_batched()`
- Import: `from src.utils.scoring import predict_prob, predict_prob_batched`

---

#### [MODIFY] [train_memae.py](file:///home/envyiu/Music/IDS2/src/models/memae/train_memae.py)

- Hợp nhất `_reconstruction_stats()` + `_reconstruction_scores()` thành:
  ```python
  def _reconstruction_errors(model, X, device, batch_size, reduction='sum'):
      # reduction='sum' → scores cho calibration (sum per sample)
      # reduction='mean' → stats cho logging (mean per sample)
  ```

---

#### [DELETE] [window_features.py](file:///home/envyiu/Music/IDS2/src/features/window_features.py)

- File shim 6 dòng, backward-compat re-export
- Kiểm tra: không có import nào từ file này ngoài window package

---

### Phase 2: MemAE Generalization Improvements

---

#### [MODIFY] [model.py](file:///home/envyiu/Music/IDS2/src/models/memae/model.py)

Thêm **Entropy Separation Loss** vào MemAE training — thay vì thay đổi kiến trúc (rủi ro cao), tăng cường loss function:

1. **Diversity regularization trên memory bank:**
   ```python
   def memory_diversity_loss(self):
       # Phạt memory slots quá giống nhau → buộc mỗi slot mã hóa pattern khác nhau
       normed = F.normalize(self.memory, dim=1)
       sim = torch.matmul(normed, normed.t())
       mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
       return sim[mask].pow(2).mean()
   ```

2. **Memory size tăng từ 128 → 256** trong config — nhiều prototype hơn để encode đa dạng pattern benign

---

#### [MODIFY] [train_memae.py](file:///home/envyiu/Music/IDS2/src/models/memae/train_memae.py)

1. Thêm `memory_diversity_weight` vào loss:
   ```python
   loss = recon + entropy_weight * entropy + diversity_weight * model.memory_diversity_loss()
   ```

2. **Cải thiện model selection** khi `val_seen_attack` quá ít (< 10 samples):
   - Fallback sang `val_loss` thay vì metric vô nghĩa thống kê
   - Log warning rõ ràng

---

#### [MODIFY] [memae_targeted.yaml](file:///home/envyiu/Music/IDS2/configs/memae_targeted.yaml)

```yaml
model:
  memory_size: 256        # 128 → 256
  # ... giữ nguyên khác

training:
  memory_diversity_weight: 0.001  # NEW
```

---

### Phase 3: XGBoost Generalization Improvements

---

#### [MODIFY] [train_feature_set.py](file:///home/envyiu/Music/IDS2/src/models/xgboost/train_feature_set.py)

1. **Thêm Feature Selection tự động dựa trên permutation importance:**
   ```python
   def _select_features(model, X_val, y_val, threshold=0.0):
       # After initial training, compute permutation importance
       # Remove features with importance <= threshold
       # Retrain on selected features
   ```

2. **Validate best_iteration > 0:**
   ```python
   if best_iteration == 0:
       logger.warning(f"XGBoost failed to learn (best_iteration=0). "
                      f"Consider adjusting hyperparameters or features.")
   ```

---

#### [MODIFY] [xgboost_zdr5.yaml](file:///home/envyiu/Music/IDS2/configs/xgboost_zdr5.yaml)

```yaml
binary_detection:
  max_depth: 6           # 8 → 6 (giảm overfitting seen families)
  gamma: 0.3             # 0.1 → 0.3 (tăng regularization)
  reg_alpha: 0.01        # 0.0005 → 0.01
  reg_lambda: 2.0        # 1.0 → 2.0
  min_child_weight: 5    # 2 → 5 (giảm fit noise)

training:
  feature_selection: true              # NEW
  feature_selection_threshold: 0.0     # NEW: remove zero-importance features
```

---

### Phase 4: Fusion Improvements

---

#### [MODIFY] [train_score_fusion.py](file:///home/envyiu/Music/IDS2/src/models/fusion/train_score_fusion.py)

1. **Thêm fusion features mới:**
   ```python
   def _fusion_features(xgb_score, memae_score):
       # ... existing 6 features ...
       # NEW features:
       abs_diff = np.abs(xgb_score - memae_score)      # disagreement signal
       min_score = np.minimum(xgb_score, tanh_memae)   # conservative agreement
       # Total: 8 features
   ```

2. **Thay LogisticRegression bằng CalibratedClassifierCV:**
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   base = LogisticRegression(class_weight='balanced', max_iter=2000)
   clf = CalibratedClassifierCV(base, cv=3, method='isotonic')
   ```

---

### Phase 5: Botnet-Specific Feature Engineering

---

#### [MODIFY] [engine.py](file:///home/envyiu/Music/IDS2/src/features/window/engine.py)

Thêm nhóm features mới cho botnet pattern detection:

1. **Beaconing regularity features:**
   ```python
   # Phát hiện C2 callback pattern: regular interval traffic
   win_inter_arrival_regularity_W  # đã có nhưng cần strengthen
   win_beaconing_score_W = 1 / (1 + inter_arrival_cv) * (flow_count / W)
   ```

2. **Destination diversity over time:**
   ```python
   # Botnet: ít dest IP, lặp lại nhiều lần
   time_dest_ip_repeat_ratio_Ts  # đã có nhưng cần check
   time_dest_ip_concentration_Ts = max_dest_ip_count / total_flow_count
   ```

---

#### [MODIFY] [names.py](file:///home/envyiu/Music/IDS2/src/features/window/names.py)

Thêm tên features mới tương ứng.

---

#### [MODIFY] [config.py](file:///home/envyiu/Music/IDS2/src/features/window/config.py)

Thêm config cho features mới:
```python
DEFAULT_WINDOW_CONFIG['include_beaconing_detection'] = True
```

---

#### [MODIFY] [window_features_zdr5.yaml](file:///home/envyiu/Music/IDS2/configs/window_features_zdr5.yaml)

```yaml
include_beaconing_detection: true  # NEW
```

---

### Phase 6: Test Updates

---

#### [MODIFY] [test_pipeline_hardening.py](file:///home/envyiu/Music/IDS2/tests/test_pipeline_hardening.py)

1. Cập nhật imports cho refactored modules
2. Thêm test cho `src.utils.scoring`:
   - `test_predict_prob_uses_best_iteration`
   - `test_threshold_for_fpr_with_jitter`
   - `test_threshold_for_fpr_without_jitter`
   - `test_metrics_from_pred_matches_original`
3. Thêm test cho `src.utils.reporting`:
   - `test_markdown_table_format`
4. Thêm test cho MemAE diversity loss
5. Thêm test cho mới fusion features (8 dims thay vì 6)
6. Update existing window feature tests cho beaconing features

---

## Execution Order

```
Phase 1 (Code Refactor) ──── ~4 files thay đổi, verification bằng existing tests
    │
    ├── 1.1 Tạo src/utils/scoring.py + reporting.py
    ├── 1.2 Refactor detector_calibration.py 
    ├── 1.3 Refactor fusion_calibration.py
    ├── 1.4 Refactor train_feature_set.py + train_score_fusion.py
    ├── 1.5 Refactor train_memae.py
    ├── 1.6 Xóa window_features.py shim
    └── 1.7 Run tests → xác nhận zero regression
    │
Phase 2 (MemAE) ──── model.py + train_memae.py + config
    │
Phase 3 (XGBoost) ──── train_feature_set.py + config
    │
Phase 4 (Fusion) ──── train_score_fusion.py
    │
Phase 5 (Botnet Features) ──── window/ package + config
    │
Phase 6 (Tests) ──── cập nhật + thêm tests mới
```

---

## Verification Plan

### Automated Tests
```bash
# Sau Phase 1: chạy tests hiện có → phải PASS 100%
.venv/bin/python -m unittest tests/test_pipeline_hardening.py -v

# Sau Phase 6: chạy tests mới
.venv/bin/python -m unittest tests/test_pipeline_hardening.py -v
```

### Full Retrain Verification
```bash
# Sau tất cả phases: full pipeline retrain
.venv/bin/python scripts/run_full_pipeline_all_families.py \
  --families all \
  --force-retrain \
  --clean-data

# So sánh kết quả với run_20260503_094559:
# - Macro Z-DR phải cải thiện (target > 0.85)
# - Worst-family Z-DR (botnet) phải cải thiện (target > 0.50)
# - Không có family nào bị regression quá 5%
```

### Manual Verification
- Review từng refactored file để đảm bảo logic không đổi (Phase 1)
- Kiểm tra `feature_schema.json` sau preprocess (Phase 5)
- So sánh `memae_feature_schema.json` trước/sau (Phase 2)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 1 refactor gây regression | Low | High | Existing tests + git diff review |
| MemAE diversity loss gây val_loss tăng | Medium | Medium | Tunable weight, có thể tắt |
| XGBoost regularization giảm seen detection | Medium | Medium | So sánh per-family trước/sau |
| New window features không help botnet | Medium | Low | Features có thể tắt qua config |
| CalibratedClassifierCV chậm hơn | Low | Low | Có thể rollback |

---

## Files Changed Summary

| Action | File | Phase |
|---|---|---|
| NEW | `src/utils/scoring.py` | 1 |
| NEW | `src/utils/reporting.py` | 1 |
| MODIFY | `src/evaluation/detector_calibration.py` | 1 |
| MODIFY | `src/evaluation/fusion_calibration.py` | 1 |
| MODIFY | `src/models/xgboost/train_feature_set.py` | 1, 3 |
| MODIFY | `src/models/fusion/train_score_fusion.py` | 1, 4 |
| MODIFY | `src/models/memae/train_memae.py` | 1, 2 |
| MODIFY | `src/models/memae/model.py` | 2 |
| MODIFY | `configs/memae_targeted.yaml` | 2 |
| MODIFY | `configs/xgboost_zdr5.yaml` | 3 |
| MODIFY | `configs/window_features_zdr5.yaml` | 5 |
| MODIFY | `src/features/window/engine.py` | 5 |
| MODIFY | `src/features/window/names.py` | 5 |
| MODIFY | `src/features/window/config.py` | 5 |
| MODIFY | `tests/test_pipeline_hardening.py` | 6 |
| DELETE | `src/features/window_features.py` | 1 |
