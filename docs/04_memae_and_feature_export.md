# Module: MemAE & Feature Export

File này mô tả sâu hai phần liền nhau trong pipeline:

```text
src/models/memae/model.py
src/models/memae/train_memae.py
src/features/export_memae_features.py
```

MemAE là detector bất thường học trên benign-only. Tuy nhiên trong pipeline này, MemAE không chỉ là model độc lập: nó còn là bộ trích xuất representation cho XGBoost và score fusion.

---

## 1. Input và output của module

Input chính:

```text
data/processed/{experiment}/
  X_train.npy
  y_train.npy
  X_val.npy
  y_val.npy
  X_test_seen.npy
  X_test_zero_day.npy
  feature_schema.json
```

Output training:

```text
artifacts/memae/{artifact_name}/
  memae_best.pt
  training_log.json
```

Output feature export:

```text
data/features/{feature_set}/
  F_train.npy
  F_val.npy
  F_test_seen.npy
  F_test_zero_day.npy
  memae_feature_schema.json
```

Trong recipe chính:

```text
experiment    = zero_day_{family}_host_disjoint_zdr5
artifact_name = zero_day_{family}_host_disjoint_zdr5_targetsel_zdr5
feature_set   = zero_day_{family}_host_disjoint_zdr5_targetsel_zdr5
```

---

## 2. Kiến trúc `MemAE`

Class chính:

```python
class MemAE(nn.Module)
```

Tham số:

```text
input_dim         # số feature processed trong X_*.npy
latent_dim        # kích thước latent z
memory_size       # số slot trong memory bank
shrink_threshold  # ngưỡng hard shrink attention
hidden_dims       # encoder/decoder hidden layers
dropout           # dropout sau ReLU nếu > 0
```

Config chính `configs/memae_targeted.yaml`:

```yaml
model:
  latent_dim: 48
  memory_size: 128
  shrink_threshold: 0.0078125
  hidden_dims: [128, 64]
  dropout: 0.1
```

### 2.1 Encoder

Encoder là MLP:

```text
input_dim
  -> Linear(input_dim, 128)
  -> BatchNorm1d
  -> ReLU
  -> Dropout(0.1)
  -> Linear(128, 64)
  -> BatchNorm1d
  -> ReLU
  -> Dropout(0.1)
  -> Linear(64, latent_dim)
```

Nếu `hidden_dims` không truyền vào, model dùng một hidden layer:

```python
hidden = max(64, min(256, input_dim * 2))
```

Trong recipe hiện tại hidden dims được khai báo rõ nên không dùng fallback này.

### 2.2 Memory bank

```python
self.memory = nn.Parameter(torch.randn(memory_size, latent_dim) * 0.05)
```

Memory là ma trận trainable kích thước:

```text
memory_size x latent_dim = 128 x 48
```

Với mỗi latent vector `z`, model tính attention:

```text
attn = softmax(z @ memory.T)
```

Sau đó dùng attention để đọc memory:

```text
z_hat = attn @ memory
```

`z_hat` là latent đã bị ép biểu diễn qua các prototype trong memory. Nếu flow khác biệt với pattern benign, reconstruction thường kém hơn.

### 2.3 Hard shrink attention

`hard_shrink_relu(weights)` áp dụng:

```text
weights = relu(weights - shrink_threshold) * weights / abs(weights - shrink_threshold)
weights = weights / sum(weights)
```

Mục đích:

- Làm attention sparse hơn.
- Giảm việc mỗi sample dùng quá nhiều memory slot.
- Buộc latent đi qua vài prototype nổi bật thay vì nội suy mịn toàn bộ memory.

Nếu `shrink_threshold <= 0`, hàm trả attention gốc.

### 2.4 Decoder

Decoder đảo chiều `hidden_dims`:

```text
latent_dim
  -> Linear(48, 64)
  -> BatchNorm1d
  -> ReLU
  -> Dropout(0.1)
  -> Linear(64, 128)
  -> BatchNorm1d
  -> ReLU
  -> Dropout(0.1)
  -> Linear(128, input_dim)
```

Forward trả về:

```python
x_hat, z, z_hat, attn = model(batch)
```

Các tensor này đều được dùng trong export feature.

---

## 3. Loss function

```python
def memae_loss(x, x_hat, attn, entropy_weight=0.0002)
```

Loss gồm:

```text
recon = mse(x_hat, x)
entropy = mean(sum(-attn * log(attn)))
loss = recon + entropy_weight * entropy
```

`recon` đo khả năng tái tạo. `entropy` phạt attention phân tán. Vì entropy được cộng với hệ số dương, training có xu hướng chọn attention ít phân tán hơn.

Trong config:

```yaml
training:
  entropy_weight: 0.0002
```

Lưu ý: training dùng MSE trung bình theo toàn batch/toàn chiều, còn scoring trong `_reconstruction_scores()` dùng tổng bình phương lỗi theo từng sample:

```python
err = ((batch - x_hat) ** 2).sum(dim=1)
```

Do đó score MemAE dùng cho calibration là reconstruction error dạng sum, không phải mean.

---

## 4. Training loop

Entry:

```python
train_memae(experiment, config, seed=42, artifact_name=None)
```

### 4.1 Benign-only training

Code load:

```python
X_train = np.load(... X_train.npy)
y_train = np.load(... y_train.npy)
X_val   = np.load(... X_val.npy)
y_val   = np.load(... y_val.npy)
```

Sau đó tách:

```text
train_benign    = X_train[y_train == 0]
val_benign      = X_val[y_val == 0]
val_seen_attack = X_val[y_val == 1]
```

Model chỉ train trên `train_benign`. `val_seen_attack` không dùng để update weight, chỉ dùng cho selection sanity nếu chọn metric `seen_recall_at_benign_fpr`.

### 4.2 Sampling

`_sample(X, max_samples, seed)` random không hoàn lại nếu số dòng vượt `max_samples`.

Config hiện tại:

```yaml
max_train_samples:
max_val_samples:
```

Hai giá trị đang để null, tức dùng toàn bộ benign train/val. Nếu dataset lớn hơn khả năng RAM/GPU, có thể đặt giới hạn tại đây.

### 4.3 Optimizer và scheduler

```text
optimizer = Adam(lr=0.0005, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(
  mode=min,
  factor=0.5,
  patience=3,
  min_lr=1e-5
)
```

Early stopping dùng `training.patience = 12`. Nếu không cải thiện theo selection metric trong 12 epoch, loop dừng.

### 4.4 Selection metric

Config chính:

```yaml
selection:
  metric: seen_recall_at_benign_fpr
  target_fpr: 0.01
```

Với metric này, mỗi epoch:

1. Tính reconstruction scores trên `val_benign`.
2. Tính reconstruction scores trên sample `val_seen_attack`.
3. Chọn threshold bằng quantile benign:

```text
threshold = quantile(benign_score, 1 - target_fpr)
```

4. Tính:

```text
val_benign_fpr = mean(benign_score >= threshold)
val_seen_attack_recall = mean(attack_score >= threshold)
```

Epoch tốt hơn nếu `val_seen_attack_recall` cao hơn. Nếu bằng nhau, epoch có `val_loss` thấp hơn thắng.

Nếu `selection.metric` không phải `seen_recall_at_benign_fpr`, code fallback về chọn `val_loss` nhỏ nhất.

### 4.5 Checkpoint

Khi epoch tốt hơn, code ghi:

```python
torch.save(
  {
    "model_state_dict": model.state_dict(),
    "input_dim": train_benign.shape[1],
    "model_config": model_cfg,
    "selection_metric": selection_metric,
  },
  artifact_dir / "memae_best.pt",
)
```

Checkpoint chứa đủ thông tin để dựng lại model khi export features. Nó không chứa optimizer state vì pipeline không resume training.

### 4.6 Training log

`training_log.json` gồm:

```text
experiment
device
best_epoch
best_val_loss
selection_metric
selection_target_fpr
best_selection_value
best_selection_summary
train_benign_samples_used
val_benign_samples_used
val_seen_attack_samples_used_for_sanity
reconstruction_stats
history
```

`reconstruction_stats` ghi mean/std/p50/p90/p95/p99 cho `val_benign` và `val_seen_attack`. Đây là nơi kiểm tra nhanh MemAE có tách được benign và seen attacks không.

---

## 5. Feature export

Entry:

```python
export_features(
  experiment,
  batch_size=4096,
  artifact_name=None,
  feature_set=None,
  include_raw_input=False,
  raw_input_feature_patterns=None,
)
```

Hàm load checkpoint:

```text
artifacts/memae/{artifact_name}/memae_best.pt
```

Sau đó dựng `MemAE(checkpoint["input_dim"], **checkpoint["model_config"])` và chạy inference cho bốn split.

### 5.1 MemAE-derived feature blocks

`_batch_features(model, batch)` tạo vector:

```text
re_scalar          # 1 chiều: sum squared reconstruction error
residual           # D chiều: x - x_hat
abs_residual       # D chiều: abs(x - x_hat)
latent_z           # C chiều
latent_z_hat       # C chiều
latent_deviation   # C chiều: z - z_hat
attn_entropy       # 1 chiều
attn_sparsity      # 1 chiều: số attention weight > 1e-4
attn_max           # 1 chiều: max attention weight
```

Với:

```text
D = input_dim
C = latent_dim
```

Số chiều MemAE-derived:

```text
1 + D + D + C + C + C + 1 + 1 + 1
= 4 + 2D + 3C
```

Code thể hiện bằng:

```python
_memae_feature_dim(input_dim, latent_dim) = 4 + 2 * input_dim + 3 * latent_dim
```

Với `latent_dim = 48`, phần latent đóng góp `144` chiều; phần residual đóng góp `2D`; 4 chiều scalar còn lại gồm reconstruction score và attention summaries.

### 5.2 Optional raw processed input

Nếu `include_raw_input=True`, export sẽ append thêm raw processed `X` vào cuối `F`.

Nếu `raw_input_feature_patterns` rỗng, append toàn bộ D chiều.

Nếu truyền pattern, `_raw_feature_indices()` chọn các feature có tên chứa một trong các pattern:

```text
feature_order[idx] contains pattern
```

Ví dụ:

```bash
--include-raw-input-features --raw-input-feature-pattern win_
```

sẽ append các processed feature có tên chứa `win_` nếu `feature_schema.json` có `feature_order`.

Nếu pattern không match feature nào, hàm raise `ValueError`.

### 5.3 Ghi memmap

`_extract_to_memmap()` dùng:

```python
open_memmap(output_path, mode="w+", dtype="float32", shape=(len(X), feature_dim))
```

Mỗi batch:

1. Chạy model trên device.
2. Tạo feature tensor.
3. Nếu cần, concatenate raw input.
4. Assert toàn bộ finite.
5. Assert số cột đúng `feature_dim`.
6. Ghi vào memmap theo offset.

Do output là `.npy` memmap chuẩn, các bước sau có thể `np.load(..., mmap_mode="r")` mà không load toàn bộ vào RAM.

### 5.4 Schema export

`memae_feature_schema.json` ghi:

```text
experiment
feature_set
artifact_name
D_value
C_value
include_raw_input
raw_input_dim
raw_input_feature_patterns
raw_input_feature_indices
raw_input_feature_names
memae_feature_dim
total_dims_numeric
processed_feature_count
processed_benchmark_mode
processed_window_features
shapes
feature_blocks
```

Đây là file quan trọng để biết `F_*.npy` đang chứa những block nào và số chiều đúng là bao nhiêu.

---

## 6. Cách MemAE được dùng sau export

Sau export, các downstream module dùng `F_*` theo hai cách:

1. XGBoost dùng toàn bộ vector `F_*` làm input.
2. Calibration và fusion dùng cột đầu tiên `F[:, 0]` làm `memae_score`.

Cột đầu tiên luôn là `re_scalar`. Đây là contract quan trọng. Nếu đổi thứ tự `_batch_features()`, toàn bộ calibration/fusion sẽ hiểu sai score MemAE.

Trong `train_score_fusion.py`:

```python
memae_train = F_train[:, 0]
memae_val = F_val[:, 0]
```

Trong evaluation:

```python
memae_val_score = np.asarray(val["F"][:, 0], dtype=np.float32)
```

Vì vậy khi mở rộng feature block, chỉ nên append vào sau, không đưa feature khác lên trước `re_scalar`.

---

## 7. Các failure mode thường gặp

### 7.1 BatchNorm với batch quá nhỏ

Model dùng `BatchNorm1d`. Training batch cuối quá nhỏ vẫn thường chạy được, nhưng nếu dataset cực nhỏ và batch size không phù hợp có thể gây thống kê không ổn định. Với CIC-IDS2017, batch size 4096 không phải vấn đề.

### 7.2 Không có benign trong train

Nếu split lỗi khiến `train_benign` rỗng, DataLoader/training sẽ fail. Đây là lỗi split nghiêm trọng, không nên xử lý im lặng.

### 7.3 Reconstruction score scale đổi khi đổi preprocessor

MemAE học trên `X_*.npy` đã standardize. Nếu thay clipping/scaler/window feature order, checkpoint cũ không còn tương thích dù input_dim có thể trùng. Cần retrain MemAE và export lại features.

### 7.4 `F[:, 0]` không còn là reconstruction error

Đây là lỗi contract nếu sửa `_batch_features()`. Downstream sẽ calibrate sai. Luôn giữ `re_scalar` ở vị trí đầu.

### 7.5 Raw input pattern không match

Nếu bật `include_raw_input=True` với pattern nhưng `feature_order` thiếu hoặc không có match, export raise lỗi. Kiểm tra `data/processed/{experiment}/feature_schema.json` trước.

---

## 8. Kiểm thử hiện có

`tests/test_pipeline_hardening.py::test_memae_export_can_append_raw_processed_input` dựng một experiment nhỏ trong tempdir:

- Tạo `X_train`, `X_val`, `X_test_seen`, `X_test_zero_day` với 3 feature.
- Tạo `feature_schema.json` có `["a", "b", "c"]`.
- Tạo checkpoint MemAE input_dim=3, latent_dim=2.
- Export plain features.
- Export features có append raw input.
- Export features chỉ append raw feature match pattern `"b"`.

Test xác nhận:

```text
plain total_dims = 4 + 2D + 3C
raw total_dims   = memae_dim + 3
selected raw_dim = 1
raw_input_feature_names = ["b"]
```

Đây là test bảo vệ contract dimension/schema cho export.

---

## 9. Checklist khi chỉnh MemAE/export

1. Nếu đổi kiến trúc, đảm bảo checkpoint vẫn ghi `input_dim` và `model_config`.
2. Nếu đổi feature export, giữ `re_scalar` là cột 0.
3. Nếu thêm block feature mới, cập nhật `feature_blocks` trong schema.
4. Nếu đổi công thức score, cập nhật detector/fusion calibration tương ứng.
5. Nếu đổi preprocessor hoặc window config, retrain MemAE trước khi export.
6. Luôn kiểm tra `memae_feature_schema.json` sau export: `shapes`, `D_value`, `C_value`, `total_dims_numeric`.
