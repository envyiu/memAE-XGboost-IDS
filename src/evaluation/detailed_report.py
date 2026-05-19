"""Comprehensive model evaluation report with publication-quality visualizations."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.utils.io import ensure_dir, read_json
from src.utils.scoring import attach_selected_feature_indices, predict_prob

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
_PALETTE = {"benign": "#2ecc71", "seen_attack": "#e67e22", "zero_day": "#e74c3c"}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.15})


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split(processed_dir: Path, feature_dir: Path, split: str):
    return {
        "F": np.load(feature_dir / f"F_{split}.npy", mmap_mode="r"),
        "y": np.load(processed_dir / f"y_{split}.npy"),
        "family": np.load(processed_dir / f"family_{split}.npy", allow_pickle=True),
    }


def _val_split_name(processed_dir: Path, feature_dir: Path) -> str:
    if (feature_dir / "F_model_selection_val.npy").exists() and (processed_dir / "y_model_selection_val.npy").exists():
        return "model_selection_val"
    return "val"


def _subsample(arrays: dict, n: int, seed: int = 42) -> dict:
    total = len(next(iter(arrays.values())))
    if total <= n:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n, replace=False)
    idx.sort()
    return {k: v[idx] if isinstance(v, np.ndarray) else v for k, v in arrays.items()}


# ---------------------------------------------------------------------------
# Section 1: Data Overview
# ---------------------------------------------------------------------------

def _section_data_overview(splits: dict[str, dict], feature_schema: dict, out: Path) -> str:
    lines = ["## 1. Data Overview\n"]
    # Split composition table
    header = "| Split | Total |"
    sep = "| --- | ---: |"
    all_families = sorted({f for s in splits.values() for f in np.unique(s["family"])})
    for f in all_families:
        header += f" {f} |"
        sep += " ---: |"
    lines += [header, sep]
    for name, data in splits.items():
        fam = data["family"]
        counts = Counter(fam)
        row = f"| {name} | {len(fam)} |"
        for f in all_families:
            row += f" {counts.get(f, 0)} |"
        lines.append(row)
    # Feature dims
    D = feature_schema.get("D_value", "?")
    total_cols = feature_schema.get("shapes", {})
    first_shape = next(iter(total_cols.values()), [0, 0]) if total_cols else [0, 0]
    n_features = first_shape[1] if len(first_shape) > 1 else "?"
    lines += [
        "",
        f"- **MemAE input dim (D):** {D}",
        f"- **Total feature columns (F):** {n_features}",
        f"- **Include raw input:** {feature_schema.get('include_raw_input', False)}",
        f"- **Feature blocks:** {', '.join(feature_schema.get('feature_blocks', []))}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: Score Distributions
# ---------------------------------------------------------------------------

def _section_score_distributions(splits: dict[str, dict], xgb_scores: dict, out: Path) -> str:
    lines = ["## 2. Score Distribution Analysis\n"]
    imgs = []
    for score_name, score_source in [("MemAE Reconstruction Error", "memae"), ("XGBoost Probability", "xgb")]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(score_name, fontsize=14, fontweight="bold")
        for ax, split_name in zip(axes, ["test_seen", "test_zero_day"]):
            data = splits[split_name]
            if score_source == "memae":
                scores = np.asarray(data["F"][:, 0], dtype=np.float32)
            else:
                scores = xgb_scores[split_name]
            benign_mask = data["family"] == "benign"
            attack_mask = ~benign_mask
            if benign_mask.any():
                ax.hist(scores[benign_mask], bins=100, alpha=0.6, label="Benign", color=_PALETTE["benign"], density=True)
            if attack_mask.any():
                label = "Zero-day" if split_name == "test_zero_day" else "Seen attack"
                color = _PALETTE["zero_day"] if split_name == "test_zero_day" else _PALETTE["seen_attack"]
                ax.hist(scores[attack_mask], bins=100, alpha=0.6, label=label, color=color, density=True)
            ax.set_title(split_name, fontsize=11)
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"dist_{score_source}.png"
        _save(fig, out / fname)
        imgs.append(fname)

    # Violin plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Score Violin Plots (test_zero_day)", fontsize=14, fontweight="bold")
    zd = splits["test_zero_day"]
    for ax, (title, scores) in zip(axes, [
        ("MemAE RE", np.asarray(zd["F"][:, 0], dtype=np.float32)),
        ("XGBoost Prob", xgb_scores["test_zero_day"]),
    ]):
        families_arr = zd["family"]
        unique_fams = sorted(set(families_arr))
        plot_data = []
        plot_labels = []
        for f in unique_fams:
            mask = families_arr == f
            s = scores[mask]
            if len(s) > 5000:
                s = np.random.default_rng(42).choice(s, 5000, replace=False)
            plot_data.append(s)
            plot_labels.append(f)
        ax.violinplot(plot_data, showmedians=True)
        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels(plot_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out / "dist_violin.png")
    imgs.append("dist_violin.png")

    for img in imgs:
        lines.append(f"![{img}]({img})\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: ROC & PR Curves
# ---------------------------------------------------------------------------

def _section_roc_pr(splits: dict, xgb_scores: dict, out: Path) -> str:
    lines = ["## 3. ROC & Precision-Recall Curves\n"]
    zd = splits["test_zero_day"]
    y = zd["y"]
    memae = np.asarray(zd["F"][:, 0], dtype=np.float32)
    xgb_s = xgb_scores["test_zero_day"]
    or_score = np.maximum(memae / (np.percentile(memae, 99) + 1e-9), xgb_s)

    models = [("MemAE", memae), ("XGBoost", xgb_s), ("OR-fusion (approx)", or_score)]
    colors = ["#3498db", "#e74c3c", "#9b59b6"]

    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, score), color in zip(models, colors):
        fpr, tpr, _ = roc_curve(y, score)
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.4f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — test_zero_day", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 0.15)
    _save(fig, out / "roc_curves.png")

    # PR
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, score), color in zip(models, colors):
        prec, rec, _ = precision_recall_curve(y, score)
        auprc = auc(rec, prec)
        ax.plot(rec, prec, label=f"{name} (AUPRC={auprc:.4f})", color=color, linewidth=2)
    ax.set_xlabel("Recall (Zero-Day Detection Rate)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — test_zero_day", fontweight="bold")
    ax.legend(loc="lower left")
    _save(fig, out / "pr_curves.png")

    lines += ["![ROC](roc_curves.png)\n", "![PR](pr_curves.png)\n"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: Confusion Matrices
# ---------------------------------------------------------------------------

def _section_confusion(splits: dict, xgb_scores: dict, threshold: float, out: Path) -> str:
    lines = ["## 4. Confusion Matrices (XGBoost @ primary threshold)\n"]
    for split_name in ["test_seen", "test_zero_day"]:
        data = splits[split_name]
        pred = (xgb_scores[split_name] >= threshold).astype(int)
        cm = confusion_matrix(data["y"], pred, labels=[0, 1])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Confusion Matrix — {split_name}", fontweight="bold")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"],
                    yticklabels=["Benign", "Attack"], ax=axes[0])
        axes[0].set_title("Absolute Counts")
        axes[0].set_ylabel("Actual")
        axes[0].set_xlabel("Predicted")
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=["Benign", "Attack"],
                    yticklabels=["Benign", "Attack"], ax=axes[1])
        axes[1].set_title("Normalized")
        axes[1].set_ylabel("Actual")
        axes[1].set_xlabel("Predicted")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"cm_{split_name}.png"
        _save(fig, out / fname)
        lines.append(f"![{split_name}]({fname})\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 5: Feature Importance
# ---------------------------------------------------------------------------

def _section_feature_importance(xgb_dir: Path, feature_schema: dict, out: Path) -> str:
    lines = ["## 5. XGBoost Feature Importance\n"]
    imp_path = xgb_dir / "feature_importance.json"
    if not imp_path.exists():
        lines.append("_Feature importance file not found._\n")
        return "\n".join(lines)
    importance = read_json(imp_path).get("gain", {})
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:50]
    if not sorted_imp:
        return "\n".join(lines + ["_No features._\n"])

    D = int(feature_schema.get("D_value", 594))
    C = int(feature_schema.get("C_value", 48))
    
    # Construct feature names mapping
    feature_names = ["memae_re_scalar"]
    for i in range(D): feature_names.append(f"memae_residual_{i}")
    for i in range(D): feature_names.append(f"memae_abs_residual_{i}")
    for i in range(C): feature_names.append(f"memae_latent_z_{i}")
    for i in range(C): feature_names.append(f"memae_latent_z_hat_{i}")
    for i in range(C): feature_names.append(f"memae_latent_deviation_{i}")
    feature_names.extend(["memae_attn_entropy", "memae_attn_sparsity", "memae_attn_max"])
    
    raw_features = feature_schema.get("raw_input_feature_names", [])
    feature_names.extend(raw_features)

    # Resolve names from fXXX format
    names = []
    for f, _ in sorted_imp:
        idx = int(f[1:])
        if idx < len(feature_names):
            names.append(feature_names[idx])
        else:
            names.append(f)
            
    values = [v for _, v in sorted_imp]
    colors_list = ["#e74c3c" if int(f[1:]) >= int(feature_schema.get("memae_feature_dim", 594)) else "#3498db" for f, _ in sorted_imp]

    fig, ax = plt.subplots(figsize=(12, 14))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors_list)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title("Top-50 Feature Importance (Blue=MemAE, Red=Raw/Window)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "feature_importance.png")

    # Pie chart
    all_gain = importance
    memae_dim = int(feature_schema.get("memae_feature_dim", 594))
    memae_gain = sum(v for f, v in all_gain.items() if int(f[1:]) < memae_dim)
    raw_gain = sum(v for f, v in all_gain.items() if int(f[1:]) >= memae_dim)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([memae_gain, raw_gain], labels=["MemAE-derived", "Raw/Window"],
           autopct="%1.1f%%", colors=["#3498db", "#e74c3c"], startangle=90)
    ax.set_title("Feature Gain Contribution", fontweight="bold")
    _save(fig, out / "feature_pie.png")

    # Table
    raw_count = sum(1 for f in all_gain if int(f[1:]) >= memae_dim)
    memae_count = sum(1 for f in all_gain if int(f[1:]) < memae_dim)
    top50_raw = sum(1 for f, _ in sorted_imp if int(f[1:]) >= memae_dim)
    lines += [
        f"- Total features used: **{len(all_gain)}**",
        f"- MemAE-derived: **{memae_count}** ({memae_gain/(memae_gain+raw_gain)*100:.1f}% gain)",
        f"- Raw/Window: **{raw_count}** ({raw_gain/(memae_gain+raw_gain)*100:.1f}% gain)",
        f"- Raw/Window in Top-50: **{top50_raw}**",
        "",
        "![importance](feature_importance.png)\n",
        "![pie](feature_pie.png)\n",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 6: Threshold Sensitivity
# ---------------------------------------------------------------------------

def _section_threshold_sensitivity(splits: dict, xgb_scores: dict, out: Path) -> str:
    lines = ["## 6. Threshold Sensitivity Analysis\n"]
    zd = splits["test_zero_day"]
    y = zd["y"]
    scores = xgb_scores["test_zero_day"]

    thresholds = np.linspace(0.01, 0.99, 200)
    metrics = {"Threshold": [], "Z-DR": [], "F1": [], "FPR": [], "Precision": []}
    benign_mask = zd["family"] == "benign"
    attack_mask = ~benign_mask
    n_benign = benign_mask.sum()
    n_attack = attack_mask.sum()

    for t in thresholds:
        pred = scores >= t
        fp = (pred & benign_mask).sum()
        tp = (pred & attack_mask).sum()
        fn = (~pred & attack_mask).sum()
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / n_attack) if n_attack > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = float(fp / n_benign) if n_benign > 0 else 0.0
        metrics["Threshold"].append(t)
        metrics["Z-DR"].append(rec)
        metrics["F1"].append(f1)
        metrics["FPR"].append(fpr)
        metrics["Precision"].append(prec)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Threshold Sensitivity — XGBoost on test_zero_day", fontweight="bold", fontsize=14)
    for ax, metric, color in zip(axes.flat, ["Z-DR", "F1", "FPR", "Precision"],
                                  ["#e74c3c", "#3498db", "#e67e22", "#2ecc71"]):
        ax.plot(metrics["Threshold"], metrics[metric], color=color, linewidth=2)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        for budget in [0.001, 0.005, 0.01, 0.05]:
            ax.axvline(x=budget, color="gray", linestyle=":", alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out / "threshold_sensitivity.png")
    lines.append("![threshold](threshold_sensitivity.png)\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 7: Cross-Family Comparison
# ---------------------------------------------------------------------------

def _section_cross_family(summary: dict, out: Path) -> str:
    lines = ["## 7. Cross-Family Comparison\n"]
    results = summary.get("results", [])
    if len(results) < 2:
        lines.append("_Only one family — skipping cross-family comparison._\n")
        return "\n".join(lines)

    families = [r["family"] for r in results]
    primary_zdr = [r["primary_result"]["test_zero_day"]["z_dr"] for r in results]
    primary_f1 = [r["primary_result"]["test_zero_day"]["f1"] for r in results]
    primary_fpr = [r["primary_result"]["test_zero_day"]["fpr"] for r in results]

    # Grouped bar
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(families))
    w = 0.25
    ax.bar(x - w, primary_zdr, w, label="Z-DR", color="#e74c3c")
    ax.bar(x, primary_f1, w, label="F1", color="#3498db")
    ax.bar(x + w, primary_fpr, w, label="FPR", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_title("Primary Model Performance per Family", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(primary_zdr):
        ax.text(i - w, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out / "cross_family_bar.png")

    # Heatmap
    metric_names = ["Z-DR", "F1", "FPR", "Precision", "Recall"]
    matrix = np.array([
        primary_zdr,
        primary_f1,
        primary_fpr,
        [r["primary_result"]["test_zero_day"]["precision"] for r in results],
        [r["primary_result"]["test_zero_day"]["recall"] for r in results],
    ])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matrix, annot=True, fmt=".3f", xticklabels=families, yticklabels=metric_names,
                cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
    ax.set_title("Performance Heatmap (Primary OR-Fusion)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "cross_family_heatmap.png")

    # Summary table
    lines += [
        "| Family | Z-DR | F1 | FPR | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        t = r["primary_result"]["test_zero_day"]
        st = r["primary_result"].get("primary_selection_status", "")
        lines.append(f"| {r['family']} | {t['z_dr']:.4f} | {t['f1']:.4f} | {t['fpr']:.4f} | {st} |")
    lines += ["", "![bar](cross_family_bar.png)\n", "![heatmap](cross_family_heatmap.png)\n"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_detailed_report(
    summary_json: str | Path,
    output_dir: str | Path | None = None,
    families: list[str] | None = None,
    sample_size: int = 100_000,
) -> Path:
    """Generate a comprehensive evaluation report from a pipeline summary JSON."""
    summary_path = Path(summary_json)
    summary = read_json(summary_path)
    output_dir = Path(output_dir) if output_dir else summary_path.parent / "detailed_report"
    ensure_dir(output_dir)

    all_results = summary.get("results", [])
    if families:
        all_results = [r for r in all_results if r.get("family") in families]

    md_sections = [
        "# Detailed Model Evaluation Report\n",
        f"- **Generated from:** `{summary_path.name}`",
        f"- **Benchmark mode:** `{summary.get('benchmark_mode', '?')}`",
        f"- **Families:** {', '.join(r.get('family', '?') for r in all_results)}",
        f"- **FPR budgets:** {summary.get('fpr_budgets', [])}",
        "",
    ]

    for result in all_results:
        experiment = result["experiment"]
        feature_set = result["feature_set"]
        family = result.get("family", experiment)
        family_dir = ensure_dir(output_dir / family)

        processed_dir = Path("data/processed") / experiment
        feature_dir = Path("data/features") / feature_set
        xgb_dir = Path("artifacts/xgboost") / feature_set
        schema_path = feature_dir / "memae_feature_schema.json"
        feature_schema = read_json(schema_path) if schema_path.exists() else {}

        print(f"[detailed_report] {family}: loading data...")
        val_name = _val_split_name(processed_dir, feature_dir)
        splits = {}
        for s in [val_name, "test_seen", "test_zero_day"]:
            raw = _load_split(processed_dir, feature_dir, s)
            splits[s] = _subsample(raw, sample_size)
        if val_name != "val":
            splits["val"] = splits[val_name]

        # Load XGBoost model and compute scores
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_dir / "xgboost_model.json")
        attach_selected_feature_indices(xgb_model, xgb_dir)
        xgb_scores = {}
        for s in splits:
            xgb_scores[s] = predict_prob(xgb_model, splits[s]["F"])

        # Primary threshold
        primary = result.get("primary_result", {})
        threshold = float(primary.get("test_zero_day", {}).get("threshold", 0.5))
        # Fallback: use XGBoost threshold from artifact
        if threshold == 0.5:
            thr_path = xgb_dir / "threshold.json"
            if thr_path.exists():
                threshold = float(read_json(thr_path).get("threshold", 0.5))

        print(f"[detailed_report] {family}: generating plots...")
        md_sections.append(f"---\n# Family: `{family}`\n")
        md_sections.append(_section_data_overview(splits, feature_schema, family_dir))
        md_sections.append(_section_score_distributions(splits, xgb_scores, family_dir))
        md_sections.append(_section_roc_pr(splits, xgb_scores, family_dir))
        md_sections.append(_section_confusion(splits, xgb_scores, threshold, family_dir))
        md_sections.append(_section_feature_importance(xgb_dir, feature_schema, family_dir))
        md_sections.append(_section_threshold_sensitivity(splits, xgb_scores, family_dir))

    # Cross-family comparison (uses summary data, no heavy loading)
    if len(all_results) >= 2:
        md_sections.append("---\n")
        md_sections.append(_section_cross_family({"results": all_results}, output_dir))

    report_path = output_dir / "detailed_report.md"
    report_path.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"[detailed_report] Report written to {report_path}")
    return report_path
