import xgboost as xgb
import json
import sys

model_path = sys.argv[1]
model = xgb.Booster()
model.load_model(model_path)

importance = model.get_score(importance_type='gain')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

raw_features_count = 0
raw_gain_sum = 0
total_gain_sum = sum(score for feat, score in importance.items())

top_100_raw = 0
print("Top 100 features:")
for i, (feat, score) in enumerate(sorted_importance[:100]):
    # Extract feature index
    f_idx = int(feat[1:])
    is_raw = f_idx >= 594
    if is_raw:
        top_100_raw += 1
    print(f"{i+1:3}. {feat:5} (Score: {score:8.4f}) {'[RAW/WINDOW]' if is_raw else '[MemAE]'}")

for feat, score in importance.items():
    f_idx = int(feat[1:])
    if f_idx >= 594:
        raw_features_count += 1
        raw_gain_sum += score

print(f"\n--- Summary for {model_path} ---")
print(f"Total features used by model: {len(importance)}")
print(f"Number of RAW features in Top 100: {top_100_raw}")
print(f"Total RAW features with non-zero importance: {raw_features_count}")
print(f"Total Gain from RAW features: {raw_gain_sum:.4f} ({(raw_gain_sum/total_gain_sum)*100:.2f}%)")
