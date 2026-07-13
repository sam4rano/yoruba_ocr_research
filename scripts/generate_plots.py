#!/usr/bin/env python3
"""
Generate publication-quality SOTA performance figures from Yorùbá OCR results.

Figures generated from real result artifacts:
1. Main model comparison (grouped bar chart for CER/WER/DER)
2. Relative error reduction vs English PP-OCR baseline
3. Bootstrap confidence intervals (point + error bar for CER/WER/DER)
4. Stratified DER by text density (line plot over quartiles)
5. Error taxonomy distribution (stacked bar chart of error categories)
6. Hard-case linguistic feature benchmark, when available

By default, missing inputs are skipped rather than replaced with illustrative
data. Use ``--allow-placeholder`` only for slide/layout mockups; never for
paper figures.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Model display name mapping
MODEL_DISPLAY = {
    "paddleocr_en_pretrained": "PaddleOCR PP-OCR (EN)",
    "paddleocrvl16_zero_shot": "PaddleOCR-VL-1.6 (Zero-Shot)",
    "glm_ocr_zero_shot": "GLM-OCR (Zero-Shot)",
    "paddleocrvl16_sft": "PaddleOCR-VL-1.6 (Fine-Tuned)",
}

MODEL_ORDER = [
    "paddleocr_en_pretrained",
    "paddleocrvl16_zero_shot",
    "glm_ocr_zero_shot",
    "paddleocrvl16_sft",
]

# Harmonious HSL-derived color palette
COLORS = {
    "paddleocr_en_pretrained": "#7f8c8d",  # Slate Gray
    "paddleocrvl16_zero_shot": "#9b59b6",     # Muted Purple
    "glm_ocr_zero_shot": "#34495e",            # Indigo
    "paddleocrvl16_sft": "#16a085",     # Vibrant Teal
}

FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf", "svg")


def load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load a CSV file if it exists and has content."""
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return None


def _to_float(value) -> float | None:
    """Convert CSV values to float, accepting blanks and percent strings."""
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw == "—":
        return None
    try:
        return float(raw.rstrip("%"))
    except ValueError:
        return None


def _pct_from_rate(value) -> float | None:
    """Convert stored metric rate to percentage."""
    number = _to_float(value)
    return None if number is None else number * 100.0


def load_main_metrics(results_dir: Path) -> pd.DataFrame | None:
    """
    Load the main comparison table for plotting.

    Preference order:
    1. ``metrics_summary.csv`` from ``scripts/compile_results.py``.
    2. ``table1_main_comparison.csv``.
    3. Raw ``metrics.csv`` latest test rows, converted to summary format.
    """
    for name in ("metrics_summary.csv", "table1_main_comparison.csv"):
        df = load_csv_safe(results_dir / name)
        if df is not None and "model_label" in df.columns:
            return df

    raw = load_csv_safe(results_dir / "metrics.csv")
    if raw is None or "model" not in raw.columns:
        return None

    rows = []
    for model in MODEL_ORDER:
        subset = raw[raw["model"] == model]
        if "split" in raw.columns:
            test_subset = subset[subset["split"] == "test"]
            if not test_subset.empty:
                subset = test_subset
        if subset.empty:
            continue
        row = subset.iloc[-1]
        if str(row.get("phantom", "")).strip().lower() == "true":
            continue
        rows.append({
            "model_label": model,
            "display_name": MODEL_DISPLAY.get(model, model),
            "cer_pct": _pct_from_rate(row.get("cer")),
            "median_cer_pct": _pct_from_rate(row.get("median_cer")),
            "micro_cer_pct": _pct_from_rate(row.get("micro_cer")),
            "wer_pct": _pct_from_rate(row.get("wer")),
            "der_pct": _pct_from_rate(row.get("der")),
            "n": row.get("n", ""),
            "der_n": row.get("der_n", ""),
        })
    return pd.DataFrame(rows) if rows else None


def setup_matplotlib():
    """Import and configure matplotlib style settings."""
    cache_root = Path(tempfile.gettempdir()) / "yoruba_ocr_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper")
        # Ensure clean text and layout styling
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
        return plt, sns
    except ImportError:
        log.error("matplotlib or seaborn not installed. Plots cannot be generated.")
        raise


def save_figure(fig, fig_dir: Path, stem: str) -> None:
    """Save a figure as paper-ready PNG plus editable vector files."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fig.savefig(fig_dir / f"{stem}.{ext}")
    log.info("Saved %s.{%s}", stem, ",".join(FIGURE_FORMATS))


def short_model_name(model: str) -> str:
    """Compact model label for axes."""
    return (
        MODEL_DISPLAY.get(model, model)
        .replace("PaddleOCR-VL-1.6", "VL-1.6")
        .replace("PaddleOCR PP-OCR", "PP-OCR")
        .replace(" (Zero-Shot)", "\nZS")
        .replace(" (Fine-Tuned)", "\nFT")
        .replace(" (EN)", "\nEN")
    )


def plot_main_comparison(df: pd.DataFrame | None, fig_dir: Path, plt, sns, *, allow_placeholder: bool):
    """Plot Figure 1: Grouped bar chart of CER, WER, and DER across models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define data
    if df is not None and "model_label" in df.columns:
        # Load real data
        plot_data = []
        for model in MODEL_ORDER:
            sub = df[df["model_label"] == model]
            if not sub.empty:
                row = sub.iloc[0]
                plot_data.append({
                    "Model": MODEL_DISPLAY.get(model, model),
                    "CER": _to_float(row.get("cer_pct")),
                    "WER": _to_float(row.get("wer_pct")),
                    "DER": _to_float(row.get("der_pct")),
                })
        plot_df = pd.DataFrame(plot_data)
    else:
        if not allow_placeholder:
            log.warning("Skipping Fig 1: metrics_summary.csv missing or malformed.")
            plt.close(fig)
            return
        # Generate clean illustrative placeholder data
        log.info("Generating illustrative placeholder data for Fig 1.")
        plot_df = pd.DataFrame([
            {"Model": MODEL_DISPLAY["paddleocr_en_pretrained"], "CER": 62.4, "WER": 84.1, "DER": 95.3},
            {"Model": MODEL_DISPLAY["paddleocrvl16_zero_shot"], "CER": 24.1, "WER": 42.5, "DER": 35.1},
            {"Model": MODEL_DISPLAY["glm_ocr_zero_shot"], "CER": 19.5, "WER": 36.2, "DER": 28.6},
            {"Model": MODEL_DISPLAY["paddleocrvl16_sft"], "CER": 9.2, "WER": 18.4, "DER": 11.3},
        ])
        
    if plot_df.empty:
        plt.close(fig)
        return
    plot_df = plot_df.dropna(subset=["CER", "WER", "DER"], how="all")
    if plot_df.empty:
        log.warning("Skipping Fig 1: no numeric CER/WER/DER values available.")
        plt.close(fig)
        return

    # Reshape for bar plot
    melted = plot_df.melt(id_vars="Model", value_vars=["CER", "WER", "DER"], var_name="Metric", value_name="Error Rate (%)")
    
    # Plot bars
    sns.barplot(data=melted, x="Model", y="Error Rate (%)", hue="Metric", palette="muted", ax=ax)
    
    # Formatting
    ax.set_title("Main Metric Performance Comparison")
    ax.set_ylabel("Error Rate (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    ax.set_ylim(0, max(melted["Error Rate (%)"].max() * 1.15, 100))
    
    # Values labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=7, color='black',
                        xytext=(0, 2),
                        textcoords='offset points')
            
    plt.tight_layout()
    save_figure(fig, fig_dir, "model_metrics_comparison")
    plt.close(fig)


def plot_relative_error_reduction(df: pd.DataFrame | None, fig_dir: Path, plt, sns, *, allow_placeholder: bool):
    """Plot relative CER/WER/DER reduction against the English PP-OCR baseline."""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    if df is not None and "model_label" in df.columns:
        rows = []
        for model in MODEL_ORDER:
            sub = df[df["model_label"] == model]
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append({
                "model": model,
                "CER": _to_float(row.get("cer_pct")),
                "WER": _to_float(row.get("wer_pct")),
                "DER": _to_float(row.get("der_pct")),
            })
        source = pd.DataFrame(rows)
    else:
        source = pd.DataFrame()

    if source.empty and allow_placeholder:
        source = pd.DataFrame([
            {"model": "paddleocr_en_pretrained", "CER": 62.4, "WER": 84.1, "DER": 95.3},
            {"model": "paddleocrvl16_zero_shot", "CER": 24.1, "WER": 42.5, "DER": 35.1},
            {"model": "glm_ocr_zero_shot", "CER": 19.5, "WER": 36.2, "DER": 28.6},
            {"model": "paddleocrvl16_sft", "CER": 9.2, "WER": 18.4, "DER": 11.3},
        ])

    baseline = source[source["model"] == "paddleocr_en_pretrained"]
    if baseline.empty:
        log.warning("Skipping Fig 5: English PP-OCR baseline row is required for relative reduction.")
        plt.close(fig)
        return

    baseline_row = baseline.iloc[0]
    improvement_rows = []
    for _, row in source.iterrows():
        model = row["model"]
        if model == "paddleocr_en_pretrained":
            continue
        for metric in ("CER", "WER", "DER"):
            base_value = baseline_row.get(metric)
            model_value = row.get(metric)
            if base_value is None or pd.isna(base_value) or not base_value:
                continue
            if model_value is None or pd.isna(model_value):
                continue
            improvement_rows.append({
                "Model": short_model_name(model),
                "Metric": metric,
                "Relative reduction (%)": ((base_value - model_value) / base_value) * 100.0,
            })

    improvement_df = pd.DataFrame(improvement_rows)
    if improvement_df.empty:
        log.warning("Skipping Fig 5: no comparable metric values available.")
        plt.close(fig)
        return

    sns.barplot(
        data=improvement_df,
        x="Model",
        y="Relative reduction (%)",
        hue="Metric",
        palette="colorblind",
        ax=ax,
    )
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.set_title("Relative Error Reduction vs English PP-OCR Baseline")
    ax.set_xlabel("")
    ax.set_ylabel("Reduction in error rate (%)")
    max_abs = max(abs(improvement_df["Relative reduction (%)"].min()), abs(improvement_df["Relative reduction (%)"].max()))
    ax.set_ylim(min(-5, improvement_df["Relative reduction (%)"].min() - 5), max(10, max_abs * 1.15))
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{height:.1f}",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=7,
                xytext=(0, 2 if height >= 0 else -2),
                textcoords="offset points",
            )
    plt.tight_layout()
    save_figure(fig, fig_dir, "relative_error_reduction")
    plt.close(fig)


def plot_bootstrap_cis(df: pd.DataFrame | None, fig_dir: Path, plt, sns, *, allow_placeholder: bool):
    """Plot Figure 2: Mean and 95% Confidence Intervals for CER, WER, DER."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    metrics = ["CER", "WER", "DER"]
    any_panel = False
    
    illustrative_data = {
        "paddleocr_en_pretrained": {"CER": (62.4, 58.1, 66.8), "WER": (84.1, 79.5, 88.6), "DER": (95.3, 91.2, 98.4)},
        "paddleocrvl16_zero_shot": {"CER": (24.1, 20.2, 28.1), "WER": (42.5, 37.8, 47.1), "DER": (35.1, 30.2, 39.8)},
        "glm_ocr_zero_shot": {"CER": (19.5, 16.1, 23.0), "WER": (36.2, 31.5, 40.8), "DER": (28.6, 24.1, 33.0)},
        "paddleocrvl16_sft": {"CER": (9.2, 6.8, 11.6), "WER": (18.4, 14.2, 22.5), "DER": (11.3, 8.5, 14.1)},
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        plot_x = []
        plot_y = []
        errors_lower = []
        errors_upper = []
        colors_list = []
        
        for model in MODEL_ORDER:
            val_found = False
            if df is not None:
                sub = df[(df["model"] == model) & (df["metric"] == metric)]
                if not sub.empty:
                    row = sub.iloc[0]
                    point = float(row["point_estimate_pct"])
                    low = float(row["ci_lower_pct"])
                    high = float(row["ci_upper_pct"])
                    val_found = True
            
            if not val_found and allow_placeholder:
                # Use illustrative mock data
                point, low, high = illustrative_data[model][metric]
            elif not val_found:
                continue
                
            plot_x.append(short_model_name(model))
            plot_y.append(point)
            errors_lower.append(point - low)
            errors_upper.append(high - point)
            colors_list.append(COLORS[model])
            
        if not plot_x:
            log.warning("Skipping %s bootstrap panel: no CI rows available.", metric)
            continue

        y_err = [errors_lower, errors_upper]
        
        # Plot point + error bars
        x_pos = np.arange(len(plot_x))
        for idx in range(len(plot_x)):
            ax.errorbar(x_pos[idx], plot_y[idx], yerr=[[errors_lower[idx]], [errors_upper[idx]]], 
                        fmt='o', color=colors_list[idx], capsize=5, elinewidth=2, markeredgewidth=2, markersize=8)
        any_panel = True
            
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_x, rotation=25, ha="right")
        ax.set_title(f"{metric} (95% CI)")
        ax.set_ylabel("Error Rate (%)" if i == 0 else "")
        ax.set_xlim(-0.5, len(plot_x) - 0.5)
        ax.set_ylim(0, max(plot_y) * 1.25)
        
    if not any_panel:
        log.warning("Skipping Fig 2: no bootstrap CI rows available.")
        plt.close(fig)
        return

    plt.suptitle("Bootstrap Resampling Metric Confidence Intervals")
    plt.tight_layout()
    save_figure(fig, fig_dir, "bootstrap_confidence_intervals")
    plt.close(fig)


def plot_stratified_density(df: pd.DataFrame | None, fig_dir: Path, plt, sns, *, allow_placeholder: bool):
    """Plot Figure 3: Line plot showing DER across text density quartiles."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    quartiles = ["q1", "q2", "q3", "q4"]
    q_labels = ["Q1 (Low)", "Q2 (Med-Low)", "Q3 (Med-High)", "Q4 (High)"]
    
    illustrative_data = {
        "paddleocr_en_pretrained": [98.1, 95.4, 94.2, 95.8],
        "paddleocrvl16_zero_shot": [30.1, 33.4, 36.2, 40.8],
        "glm_ocr_zero_shot": [22.4, 26.1, 29.8, 35.2],
        "paddleocrvl16_sft": [7.8, 9.5, 12.1, 15.6],
    }
    
    for model in MODEL_ORDER:
        vals = []
        data_found = False
        if df is not None:
            sub = df[df["model"] == model]
            if not sub.empty:
                # Group by density quartile and sort
                sub_sorted = sub.sort_values(by="density_quartile")
                # We expect q1-q4
                q_vals = {row["density_quartile"]: float(row["der_pct"]) for _, row in sub_sorted.iterrows() if not pd.isna(row["der_pct"])}
                if all(q in q_vals for q in quartiles):
                    vals = [q_vals[q] for q in quartiles]
                    data_found = True
        
        if not data_found and allow_placeholder:
            vals = illustrative_data[model]
        elif not data_found:
            continue
            
        ax.plot(q_labels, vals, marker='o', linewidth=2, color=COLORS[model], label=MODEL_DISPLAY[model])

    if not ax.lines:
        log.warning("Skipping Fig 3: no complete stratified DER quartile rows available.")
        plt.close(fig)
        return
        
    ax.set_title("Stratified Diacritic Error Rate (DER) by Character Density")
    ax.set_ylabel("Diacritic Error Rate (DER %)")
    ax.set_xlabel("Diacritic Density Quartile")
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    save_figure(fig, fig_dir, "stratified_der_by_density")
    plt.close(fig)


def plot_error_taxonomy(df: pd.DataFrame | None, fig_dir: Path, plt, sns, *, allow_placeholder: bool):
    """Plot Figure 4: Stacked bar chart showing character error distribution per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    categories = ["exact_diacritics", "substitution", "deletion_heavy", "insertion_heavy", "total_tone_drop"]
    cat_display = {
        "exact_diacritics": "Exact Match",
        "substitution": "Substitution",
        "deletion_heavy": "Deletion Heavy",
        "insertion_heavy": "Insertion Heavy",
        "total_tone_drop": "Total Tone Drop"
    }
    
    illustrative_data = {
        "paddleocr_en_pretrained": [1.2, 5.4, 2.1, 0.5, 90.8],
        "paddleocrvl16_zero_shot": [62.4, 15.8, 12.1, 7.3, 2.4],
        "glm_ocr_zero_shot": [69.5, 12.4, 10.2, 6.1, 1.8],
        "paddleocrvl16_sft": [88.2, 5.3, 4.1, 2.1, 0.3]
    }
    
    plot_data = []
    
    for model in MODEL_ORDER:
        data_found = False
        vals = {}
        if df is not None:
            sub = df[df["model"] == model]
            if not sub.empty:
                # Sum the counts and convert to percentages
                total = sub["count"].sum()
                if total > 0:
                    for cat in categories:
                        count = sub[sub["category"] == cat]["count"].sum()
                        vals[cat] = (count / total) * 100
                    data_found = True
                    
        if not data_found and allow_placeholder:
            for idx, cat in enumerate(categories):
                vals[cat] = illustrative_data[model][idx]
        elif not data_found:
            continue
                
        plot_data.append({
            "Model": MODEL_DISPLAY[model],
            **{cat_display[cat]: vals[cat] for cat in categories}
        })
        
    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        log.warning("Skipping Fig 4: no error taxonomy rows available.")
        plt.close(fig)
        return
    
    # Set display columns
    display_cols = [cat_display[c] for c in categories]
    
    # Plot stacked bar chart
    plot_df.set_index("Model")[display_cols].plot(kind="bar", stacked=True, ax=ax, 
                                                 color=["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#95a5a6"])
    
    ax.set_title("Character Diacritic Error Taxonomy Distribution")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    save_figure(fig, fig_dir, "error_taxonomy_distribution")
    plt.close(fig)


def plot_hard_cases(df: pd.DataFrame | None, fig_dir: Path, plt, sns, *, allow_placeholder: bool):
    """Plot Figure 5: Grouped bar chart of CER across linguistic hard-case categories."""
    FEATURES = [
        "Named Entities",
        "Numerics",
        "Historical Orthography",
        "Code-Mixed (Yor\u00f9b\u00e1\u2013English)",
    ]
    FEATURE_SHORT = {
        "Named Entities": "Named\nEntities",
        "Numerics": "Numerics",
        "Historical Orthography": "Historical\nOrthography",
        "Code-Mixed (Yor\u00f9b\u00e1\u2013English)": "Code-Mixed\n(Yor\u00f9b\u00e1-EN)",
    }
    # Illustrative fallback values (CER %) per feature per model
    ILLUS = {
        "Named Entities":      [72.1, 42.3, 31.4, 8.2],
        "Numerics":            [68.5, 38.9, 28.7, 7.6],
        "Historical Orthography": [85.3, 55.7, 44.2, 14.9],
        "Code-Mixed (Yor\u00f9b\u00e1\u2013English)": [78.4, 47.2, 36.8, 11.3],
    }

    n_models = len(MODEL_ORDER)
    n_features = len(FEATURES)
    x = range(n_features)
    width = 0.15
    offsets = [(i - n_models / 2 + 0.5) * width for i in range(n_models)]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for m_idx, model in enumerate(MODEL_ORDER):
        cer_vals = []
        for feat in FEATURES:
            val = None
            if df is not None:
                sub = df[(df["model"] == model) & (df["feature"] == feat)]
                if not sub.empty:
                    raw = sub["cer_pct"].values[0]
                    try:
                        val = float(raw)
                    except (TypeError, ValueError):
                        val = None
            if val is None and allow_placeholder:
                feat_idx = FEATURES.index(feat)
                val = ILLUS[feat][m_idx]
            elif val is None:
                continue
            cer_vals.append(val)

        if len(cer_vals) != n_features:
            continue

        positions = [xi + offsets[m_idx] for xi in x]
        bars = ax.bar(
            positions,
            cer_vals,
            width=width * 0.9,
            label=MODEL_DISPLAY[model],
            color=colors[m_idx],
            alpha=0.87,
        )
        # Value labels on bars
        for bar, v in zip(bars, cer_vals):
            if v > 2:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )

    if not ax.patches:
        log.warning("Skipping Fig 5: no complete hard-case rows available.")
        plt.close(fig)
        return

    ax.set_xticks(list(x))
    ax.set_xticklabels([FEATURE_SHORT[f] for f in FEATURES], fontsize=10)
    ax.set_ylabel("CER (%)")
    ax.set_title("Hard-Case Benchmark — CER by Linguistic Feature Category")
    ax.set_ylim(0, max(max(v for v in ILLUS[f]) for f in FEATURES) * 1.25)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, fig_dir, "hard_cases_benchmark")
    plt.close(fig)


def main():
    """Parse args and generate plots."""
    parser = argparse.ArgumentParser(description="Generate SOTA figures for Yor\u00f9b\u00e1 OCR analysis.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory containing metrics and CSV reports.",
    )
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Generate illustrative placeholder figures when inputs are missing. Do not use for paper outputs.",
    )
    args = parser.parse_args()
    
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load inputs
    df_metrics = load_main_metrics(args.results_dir)
    df_cis = load_csv_safe(args.results_dir / "bootstrap_metric_cis.csv")
    df_density = load_csv_safe(args.results_dir / "stratified_der_by_density.csv")
    df_taxonomy = load_csv_safe(args.results_dir / "error_taxonomy.csv")
    df_features = load_csv_safe(args.results_dir / "stratified_by_linguistic_features.csv")
    
    # Initialize plotting environment
    plt, sns = setup_matplotlib()
    
    # Plot figures
    plot_main_comparison(df_metrics, fig_dir, plt, sns, allow_placeholder=args.allow_placeholder)
    plot_relative_error_reduction(df_metrics, fig_dir, plt, sns, allow_placeholder=args.allow_placeholder)
    plot_bootstrap_cis(df_cis, fig_dir, plt, sns, allow_placeholder=args.allow_placeholder)
    plot_stratified_density(df_density, fig_dir, plt, sns, allow_placeholder=args.allow_placeholder)
    plot_error_taxonomy(df_taxonomy, fig_dir, plt, sns, allow_placeholder=args.allow_placeholder)
    plot_hard_cases(df_features, fig_dir, plt, sns, allow_placeholder=args.allow_placeholder)
    
    log.info("Figure generation pass complete under %s. Warnings indicate skipped missing inputs.", fig_dir)


if __name__ == "__main__":
    main()
