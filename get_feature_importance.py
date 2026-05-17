import xgboost as xgb
import json
import sys

model_path = sys.argv[1]
model = xgb.Booster()
model.load_model(model_path)

importance = model.get_score(importance_type='gain')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print(f"Top 100 features for {model_path}:")
raw_features = 0
for i, (feat, score) in enumerate(sorted_importance[:100]):
    is_raw = not feat.startswith('memae_')
    if is_raw:
        raw_features += 1
    print(f"{i+1:3}. {feat:50} (Score: {score:.4f}) {'[RAW/WINDOW]' if is_raw else ''}")

print(f"\nSummary: Out of top 100 features, {raw_features} are RAW/WINDOW features.")
