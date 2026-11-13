"""
Range and Comparative Statistics for Brain Tumor Segmentation Pipelines.

Supports:
  [1] Pipeline 1.0 (Thresholding) vs Ground Truth
  [2] Pipeline 2.0 (Watershed) vs Ground Truth
  [3] Pipeline 3.0 (Hybrid) vs Ground Truth
  [4] All 3 Pipelines vs Ground Truth + Comprehensive Tri-Pipeline Comparative Analysis

Individual pipeline outputs -> Results/<Pipeline>/Range_Statistics/<range>/
Comparative outputs        -> Results/Comparative_Analysis/<range>/
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# ============================================================
# CONFIGURATION
# ============================================================

PATIENT_PREFIX       = "BraTS20_Training_"
PATIENT_NUMBER_WIDTH = 3
EVALUATION_REGION    = "WT"

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
# shared/ lives directly under the project root
PROJECT_DIR = os.path.dirname(_THIS_DIR)

PIPELINES = {
    1: {
        "id":                 "pipeline_1",
        "name":               "Pipeline 1.0 (Thresholding)",
        "short_name":         "Thresholding",
        "tag":                "thresholding",
        "color":              "#1f77b4",
        "folder_name":        "Pipeline_1_Thresholding",
        "tables_dir":         os.path.join(PROJECT_DIR, "Results", "Pipeline_1_Thresholding", "Tables"),
        "fallback_tables_dir":os.path.join(PROJECT_DIR, "Results", "Tables"),
        "stats_dir":          os.path.join(PROJECT_DIR, "Results", "Pipeline_1_Thresholding", "Range_Statistics"),
    },
    2: {
        "id":                 "pipeline_2",
        "name":               "Pipeline 2.0 (Watershed)",
        "short_name":         "Watershed",
        "tag":                "watershed",
        "color":              "#2ca02c",
        "folder_name":        "Pipeline_2_Watershed",
        "tables_dir":         os.path.join(PROJECT_DIR, "Results", "Pipeline_2_Watershed", "Tables"),
        "fallback_tables_dir":None,
        "stats_dir":          os.path.join(PROJECT_DIR, "Results", "Pipeline_2_Watershed", "Range_Statistics"),
    },
    3: {
        "id":                 "pipeline_3",
        "name":               "Pipeline 3.0 (Hybrid)",
        "short_name":         "Hybrid",
        "tag":                "hybrid",
        "color":              "#ff7f0e",
        "folder_name":        "Pipeline_3_Hybrid",
        "tables_dir":         os.path.join(PROJECT_DIR, "Results", "Pipeline_3_Hybrid", "Tables"),
        "fallback_tables_dir":None,
        "stats_dir":          os.path.join(PROJECT_DIR, "Results", "Pipeline_3_Hybrid", "Range_Statistics"),
    },
}

COMPARATIVE_ROOT = os.path.join(PROJECT_DIR, "Results", "Comparative_Analysis")

# ── quality thresholds ──────────────────────────────────────
DICE_EXCELLENT    = 0.85
DICE_GOOD         = 0.70
DICE_MODERATE     = 0.50
FAILURE_MARGIN    = 0.10

DICE_THRESHOLDS      = [0.50, 0.70, 0.80, 0.85, 0.90]
HD95_THRESHOLDS_MM   = [5, 10, 20]
BOOTSTRAP_SAMPLES    = 2000

# ── metric groups ────────────────────────────────────────────
GROUPS = {
    "Overlap": [
        "dice", "dice_percentage", "iou_jaccard",
        "iou_percentage", "bounding_box_iou",
    ],
    "Classification": [
        "precision", "sensitivity_recall", "specificity",
        "false_positive_rate", "false_negative_rate", "voxel_accuracy",
    ],
    "Confusion counts": [
        "true_positive_voxels", "true_negative_voxels",
        "false_positive_voxels", "false_negative_voxels",
    ],
    "Volume": [
        "predicted_voxel_count", "ground_truth_voxel_count",
        "predicted_volume_mm3", "ground_truth_volume_mm3",
        "predicted_volume_cm3", "ground_truth_volume_cm3",
        "absolute_volume_difference_mm3", "relative_volume_error",
        "volume_similarity",
        "signed_volume_difference_mm3", "signed_relative_volume_error",
    ],
    "Surface distance": ["hd95_mm", "assd_mm"],
}

IDENTITY_COLUMNS = ["patient_id", "run_id", "region"]

OUTLIER_METRICS = [
    "dice", "iou_jaccard", "precision", "sensitivity_recall",
    "specificity", "false_positive_rate", "false_negative_rate",
    "hd95_mm", "assd_mm", "relative_volume_error", "volume_similarity",
]

CORRELATION_METRICS = [
    "dice", "iou_jaccard", "precision", "sensitivity_recall",
    "specificity", "false_positive_rate", "false_negative_rate",
    "voxel_accuracy", "bounding_box_iou",
    "hd95_mm", "assd_mm",
    "ground_truth_volume_mm3", "predicted_volume_mm3",
    "relative_volume_error", "volume_similarity",
]

# metrics shown in the comparative radar / effect-size charts
COMPARATIVE_METRICS = [
    "dice", "iou_jaccard", "precision", "sensitivity_recall",
    "specificity", "bounding_box_iou", "volume_similarity",
    "hd95_mm", "assd_mm",
    "relative_volume_error",
]

# pipeline colour palette
P_COLORS = {
    1: "#1f77b4",   # blue
    2: "#2ca02c",   # green
    3: "#ff7f0e",   # orange
}
P_LABELS = {
    1: "P1: Thresholding",
    2: "P2: Watershed",
    3: "P3: Hybrid",
}


# ============================================================
# HELPERS
# ============================================================

def make_patient_id(number: int) -> str:
    return f"{PATIENT_PREFIX}{number:0{PATIENT_NUMBER_WIDTH}d}"


def ask_integer(prompt: str) -> int:
    while True:
        value = input(prompt).strip()
        try:
            num = int(value)
            if num >= 1:
                return num
            print("  Please enter a number of 1 or more.")
        except ValueError:
            print("  Please enter a whole number.")


def fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def save_figure(fig, folder: str, name: str, dpi: int = 150):
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d effect size."""
    diff = a - b
    n = len(diff)
    if n < 2:
        return np.nan
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


# ── Large-dataset axis helper ────────────────────────────────
def _sparse_xticks(ax, labels, max_ticks: int = 40):
    """
    When there are many patients, only show every k-th label so the
    x-axis stays readable.  Always includes the first and last.
    """
    n = len(labels)
    if n <= max_ticks:
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    else:
        step = max(1, n // max_ticks)
        ticks = list(range(0, n, step))
        if (n - 1) not in ticks:
            ticks.append(n - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)


def _fig_width(n: int, per_patient: float = 0.22, minimum: float = 12.0) -> float:
    """Scale figure width with patient count, capped at 60 inches."""
    return min(60.0, max(minimum, per_patient * n + minimum * 0.5))


# ============================================================
# CSV PARSING & LOADING
# ============================================================

def read_metric_csv(path: str):
    try:
        raw = pd.read_csv(path, skipinitialspace=True, dtype=str)
    except Exception:
        return None
    if raw.shape[1] < 2:
        return None
    raw = raw.iloc[:, :2]
    raw.columns = ["Metric", "Value"]
    raw = raw.dropna().apply(lambda s: s.str.strip())
    data = dict(zip(raw["Metric"], raw["Value"]))
    if "dice" not in data:
        return None
    return data


def find_evaluation_csv(patient_id: str, pipeline_config: dict):
    search_dirs = [pipeline_config["tables_dir"]]
    if pipeline_config.get("fallback_tables_dir"):
        search_dirs.append(pipeline_config["fallback_tables_dir"])

    for root_dir in search_dirs:
        table_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(table_dir):
            continue
        patterns = [
            f"{patient_id}_{EVALUATION_REGION}_evaluation_{pipeline_config['tag']}.csv",
            f"{patient_id}_{EVALUATION_REGION}_evaluation.csv",
        ]
        for pat in patterns:
            candidate = os.path.join(table_dir, pat)
            if os.path.isfile(candidate):
                data = read_metric_csv(candidate)
                if data is not None:
                    return candidate, data, ""

        csv_files = [
            os.path.join(table_dir, name)
            for name in os.listdir(table_dir)
            if name.lower().endswith(".csv") and "morphological" not in name.lower()
        ]
        csv_files.sort(key=os.path.getmtime, reverse=True)
        for path in csv_files:
            data = read_metric_csv(path)
            if data is not None and data.get("region", EVALUATION_REGION) == EVALUATION_REGION:
                return path, data, f"used {os.path.basename(path)}"

    return None, None, "no evaluation CSV found"


def load_range(start: int, end: int, pipeline_config: dict):
    records, missing, notes = [], [], []
    for number in range(start, end + 1):
        patient_id = make_patient_id(number)
        path, data, note = find_evaluation_csv(patient_id, pipeline_config)
        if data is None:
            missing.append((patient_id, note))
            continue
        if note:
            notes.append((patient_id, note))
        row = dict(data)
        row["patient_id"]   = row.get("patient_id", patient_id)
        row["source_file"]  = os.path.basename(path)
        records.append(row)
    return records, missing, notes


# ============================================================
# TABLE BUILDING
# ============================================================

def quality_tier(dice: float) -> str:
    if pd.isna(dice):
        return "No data"
    if dice >= DICE_EXCELLENT:
        return "Excellent"
    if dice >= DICE_GOOD:
        return "Good"
    if dice >= DICE_MODERATE:
        return "Moderate"
    return "Poor"


def failure_mode(row) -> str:
    dice      = row.get("dice")
    precision = row.get("precision")
    recall    = row.get("sensitivity_recall")
    if pd.isna(dice):
        return "No data"
    if dice >= DICE_EXCELLENT:
        return "None (excellent overlap)"
    if pd.isna(precision) or pd.isna(recall):
        return "Unknown"
    if recall - precision > FAILURE_MARGIN:
        return "Over-segmentation"
    if precision - recall > FAILURE_MARGIN:
        return "Under-segmentation"
    return "Mixed (boundary errors)"


def build_patient_table(records: list):
    df = pd.DataFrame(records)
    text_columns = IDENTITY_COLUMNS + ["source_file"]
    inf_counts = {}

    for col in df.columns:
        if col in text_columns:
            continue
        values  = pd.to_numeric(df[col], errors="coerce")
        n_inf   = int(np.isinf(values).sum())
        if n_inf:
            inf_counts[col] = n_inf
        df[col] = values.replace([np.inf, -np.inf], np.nan)

    df["patient_number"] = df["patient_id"].str.split("_").str[-1]

    # derived volume columns
    if {"predicted_volume_mm3", "ground_truth_volume_mm3"} <= set(df.columns):
        df["signed_volume_difference_mm3"] = (
            df["predicted_volume_mm3"] - df["ground_truth_volume_mm3"]
        )
        gt = df["ground_truth_volume_mm3"].replace(0, np.nan)
        df["signed_relative_volume_error"] = df["signed_volume_difference_mm3"] / gt

    df["quality_tier"] = df["dice"].apply(quality_tier)
    df["failure_mode"] = df.apply(failure_mode, axis=1)
    return df, inf_counts


def numeric_metric_columns(df: pd.DataFrame):
    skip    = set(IDENTITY_COLUMNS + ["source_file", "patient_number", "quality_tier", "failure_mode"])
    columns = [c for c in df.columns if c not in skip]
    ordered = []
    for names in GROUPS.values():
        ordered += [n for n in names if n in columns]
    ordered += [c for c in columns if c not in ordered]
    return ordered


def group_of(metric: str) -> str:
    for group, names in GROUPS.items():
        if metric in names:
            return group
    return "Other"


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

def bootstrap_ci(values, rng):
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, n), replace=True)
    means   = samples.mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarize_metrics(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    rng  = np.random.default_rng(0)
    rows = []
    for metric in metrics:
        series = df[metric]
        x      = series.dropna().to_numpy(dtype=float)
        n      = len(x)
        row    = {
            "metric":    metric,
            "group":     group_of(metric),
            "n_valid":   n,
            "n_missing": int(series.isna().sum()),
        }
        if n == 0:
            rows.append(row)
            continue
        mean             = x.mean()
        std              = x.std(ddof=1) if n > 1 else np.nan
        q1, median, q3   = np.percentile(x, [25, 50, 75])
        ci_low, ci_high  = bootstrap_ci(x, rng)
        # normality test (Shapiro if n<=50, D'Agostino otherwise)
        if n >= 8:
            if n <= 50:
                _, p_norm = stats.shapiro(x)
            else:
                _, p_norm = stats.normaltest(x)
        else:
            p_norm = np.nan
        row.update({
            "mean":        mean,
            "std":         std,
            "sem":         std / np.sqrt(n) if n > 1 else np.nan,
            "ci95_low":    ci_low,
            "ci95_high":   ci_high,
            "min":         x.min(),
            "q1":          q1,
            "median":      median,
            "q3":          q3,
            "max":         x.max(),
            "iqr":         q3 - q1,
            "range":       x.max() - x.min(),
            "cv":          std / abs(mean) if n > 1 and mean != 0 else np.nan,
            "skewness":    float(stats.skew(x)) if n >= 3 else np.nan,
            "kurtosis":    float(stats.kurtosis(x)) if n >= 4 else np.nan,
            "p_normality": float(p_norm),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_metrics(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"true_positive_voxels", "true_negative_voxels",
              "false_positive_voxels", "false_negative_voxels"}
    if not needed <= set(df.columns):
        return pd.DataFrame(columns=["metric", "value"])

    tp = df["true_positive_voxels"].sum()
    tn = df["true_negative_voxels"].sum()
    fp = df["false_positive_voxels"].sum()
    fn = df["false_negative_voxels"].sum()

    def r(a, b):
        return a / b if b else np.nan

    rows = [
        ("pooled_tp_voxels",          tp),
        ("pooled_tn_voxels",          tn),
        ("pooled_fp_voxels",          fp),
        ("pooled_fn_voxels",          fn),
        ("pooled_dice",               r(2*tp, 2*tp + fp + fn)),
        ("pooled_iou_jaccard",        r(tp, tp + fp + fn)),
        ("pooled_precision",          r(tp, tp + fp)),
        ("pooled_sensitivity",        r(tp, tp + fn)),
        ("pooled_specificity",        r(tn, tn + fp)),
        ("pooled_false_positive_rate",r(fp, fp + tn)),
        ("pooled_false_negative_rate",r(fn, fn + tp)),
        ("pooled_voxel_accuracy",     r(tp + tn, tp + tn + fp + fn)),
        ("pooled_f1",                 r(2*tp, 2*tp + fp + fn)),           # same as dice
        ("pooled_f2",                 r(5*tp, 5*tp + 4*fn + fp)),          # recall-weighted F
        ("pooled_mcc",                _mcc(tp, tn, fp, fn)),
    ]

    if {"predicted_volume_mm3", "ground_truth_volume_mm3"} <= set(df.columns):
        pred = df["predicted_volume_mm3"].sum()
        gt   = df["ground_truth_volume_mm3"].sum()
        rows += [
            ("total_predicted_volume_mm3",        pred),
            ("total_ground_truth_volume_mm3",      gt),
            ("pooled_signed_relative_volume_error",r(pred - gt, gt)),
        ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _mcc(tp, tn, fp, fn):
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn - fp*fn) / denom if denom else np.nan


def pass_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows   = []
    n_dice = int(df["dice"].notna().sum())
    for t in DICE_THRESHOLDS:
        count = int((df["dice"] >= t).sum())
        rows.append({"criterion": f"Dice >= {t:.2f}",
                     "patients_passing": count, "patients_evaluated": n_dice,
                     "pass_rate_percent": 100 * count / n_dice if n_dice else np.nan})
    if "hd95_mm" in df.columns:
        n_hd = int(df["hd95_mm"].notna().sum())
        for t in HD95_THRESHOLDS_MM:
            count = int((df["hd95_mm"] <= t).sum())
            rows.append({"criterion": f"HD95 <= {t} mm",
                         "patients_passing": count, "patients_evaluated": n_hd,
                         "pass_rate_percent": 100 * count / n_hd if n_hd else np.nan})
    # additional pass-rate criteria
    for col, label, op, thresh in [
        ("iou_jaccard",          "IoU >= 0.70",          ">=", 0.70),
        ("sensitivity_recall",   "Sensitivity >= 0.80",  ">=", 0.80),
        ("precision",            "Precision >= 0.70",    ">=", 0.70),
        ("volume_similarity",    "VolSim >= 0.80",        ">=", 0.80),
        ("assd_mm",              "ASSD <= 10 mm",         "<=", 10.0),
    ]:
        if col not in df.columns:
            continue
        n_col = int(df[col].notna().sum())
        if op == ">=":
            count = int((df[col] >= thresh).sum())
        else:
            count = int((df[col] <= thresh).sum())
        rows.append({"criterion": label,
                     "patients_passing": count, "patients_evaluated": n_col,
                     "pass_rate_percent": 100 * count / n_col if n_col else np.nan})
    return pd.DataFrame(rows)


def find_outliers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in OUTLIER_METRICS:
        if metric not in df.columns:
            continue
        x = df[metric].dropna()
        if len(x) < 4:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr     = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        for idx in x.index:
            val = df.loc[idx, metric]
            if val < low or val > high:
                rows.append({"metric": metric, "patient_id": df.loc[idx, "patient_id"],
                             "value": val, "lower_fence": low, "upper_fence": high,
                             "direction": "low" if val < low else "high"})
    return pd.DataFrame(rows, columns=["metric", "patient_id", "value",
                                        "lower_fence", "upper_fence", "direction"])


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CORRELATION_METRICS if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].rank().corr()


def failure_mode_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["failure_mode"].value_counts()
    total  = counts.sum()
    out    = counts.rename_axis("failure_mode").reset_index(name="patients")
    out["percent"]        = 100 * out["patients"] / total
    out["patient_numbers"] = out["failure_mode"].apply(
        lambda m: ", ".join(df.loc[df["failure_mode"] == m, "patient_number"]))
    return out


def quality_tier_table(df: pd.DataFrame) -> pd.DataFrame:
    order  = ["Excellent", "Good", "Moderate", "Poor", "No data"]
    counts = df["quality_tier"].value_counts()
    rows   = []
    for tier in order:
        n = int(counts.get(tier, 0))
        rows.append({"quality_tier": tier, "patients": n,
                     "percent": 100 * n / len(df),
                     "patient_numbers": ", ".join(
                         df.loc[df["quality_tier"] == tier, "patient_number"])})
    return pd.DataFrame(rows)


# ============================================================
# INDIVIDUAL PIPELINE GRAPHS   (10 charts)
# ============================================================

def _plot_individual_pipeline(df: pd.DataFrame, summary: pd.DataFrame,
                               p_config: dict, graph_dir: str):
    """Generate all per-pipeline diagnostic charts."""

    color  = p_config["color"]
    name   = p_config["name"]
    n      = len(df)
    S      = summary.set_index("metric")

    def _stat(m, c):
        try:
            return float(S.loc[m, c])
        except Exception:
            return np.nan

    # ── helper: sorted index (by dice descending) ───────────
    sorted_df = df.sort_values("dice", ascending=False).reset_index(drop=True)
    slabels   = sorted_df["patient_number"].tolist()
    fw        = _fig_width(n)

    # ----------------------------------------------------------
    # 01  Dice + IoU  (line plot for n>30, bar for small n)
    # ----------------------------------------------------------
    if {"dice", "iou_jaccard"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(fw, 5))
        x = np.arange(n)
        if n > 30:
            ax.plot(x, sorted_df["dice"],       color=color,    lw=1.5, label="Dice",    marker=".", markersize=4)
            ax.plot(x, sorted_df["iou_jaccard"],color="#aaaaaa", lw=1.2, label="IoU",    marker=".", markersize=4)
            ax.fill_between(x, sorted_df["dice"], alpha=0.15, color=color)
        else:
            ax.bar(x - 0.2, df["dice"],        0.4, label="Dice", color=color)
            ax.bar(x + 0.2, df["iou_jaccard"], 0.4, label="IoU",  color="#aaaaaa")
        ax.axhline(_stat("dice", "mean"),   color="crimson",  ls="--", lw=1.4,
                   label=f"Mean Dice = {_stat('dice','mean'):.3f}")
        ax.axhline(_stat("dice", "median"), color="darkblue", ls=":",  lw=1.2,
                   label=f"Median Dice = {_stat('dice','median'):.3f}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"{name}: Dice & IoU per Patient (sorted by Dice)")
        ax.legend(fontsize=8)
        _sparse_xticks(ax, slabels)
        save_figure(fig, graph_dir, "01_dice_iou_per_patient.png")

    # ----------------------------------------------------------
    # 02  Six classification metrics — violin
    # ----------------------------------------------------------
    cls_metrics = [m for m in
                   ["dice", "iou_jaccard", "precision", "sensitivity_recall",
                    "specificity", "voxel_accuracy", "bounding_box_iou",
                    "volume_similarity", "false_positive_rate", "false_negative_rate"]
                   if m in df.columns]
    if cls_metrics:
        fig, ax = plt.subplots(figsize=(max(10, len(cls_metrics) * 1.4), 6))
        data_vio = [df[m].dropna().values for m in cls_metrics]
        parts    = ax.violinplot(data_vio, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(cls_metrics) + 1))
        ax.set_xticklabels([m.replace("_", "\n") for m in cls_metrics], fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("Value")
        ax.set_title(f"{name}: Classification & Overlap Metrics Distribution")
        ax.axhline(0.70, color="grey", ls=":", lw=1, label="0.70 reference")
        ax.legend(fontsize=8)
        save_figure(fig, graph_dir, "02_classification_metrics_violin.png")

    # ----------------------------------------------------------
    # 03  Precision vs Recall scatter coloured by quality tier
    # ----------------------------------------------------------
    if {"precision", "sensitivity_recall"} <= set(df.columns):
        tier_colors = {"Excellent": "#2ca02c", "Good": "#1f77b4",
                       "Moderate":  "#ff7f0e", "Poor": "#d62728", "No data": "#aaaaaa"}
        fig, ax = plt.subplots(figsize=(7, 7))
        for tier, grp in df.groupby("quality_tier"):
            ax.scatter(grp["sensitivity_recall"], grp["precision"],
                       color=tier_colors.get(tier, "#aaaaaa"),
                       label=tier, s=50, alpha=0.75, edgecolors="white", lw=0.4)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Precision = Recall")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title(f"{name}: Precision vs Recall  (colour = quality tier)")
        ax.legend(fontsize=8)
        save_figure(fig, graph_dir, "03_precision_recall_quality.png")

    # ----------------------------------------------------------
    # 04  HD95 + ASSD distribution — violin + strip
    # ----------------------------------------------------------
    dist_metrics = [m for m in ["hd95_mm", "assd_mm"] if m in df.columns]
    if dist_metrics:
        fig, axes = plt.subplots(1, len(dist_metrics),
                                  figsize=(5 * len(dist_metrics), 6), sharey=False)
        if len(dist_metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, dist_metrics):
            vals = df[m].dropna().values
            parts = ax.violinplot([vals], showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.ones(len(vals)) + jitter, vals,
                       color=color, alpha=0.5, s=15, zorder=3)
            ax.set_xticks([1])
            ax.set_xticklabels([m])
            ax.set_ylabel("mm")
            ax.set_title(f"{name}: {m} Distribution")
            mean_v = np.nanmean(vals)
            ax.axhline(mean_v, color="crimson", ls="--", lw=1.3,
                       label=f"Mean = {mean_v:.2f} mm")
            ax.legend(fontsize=8)
        save_figure(fig, graph_dir, "04_surface_distance_distribution.png")

    # ----------------------------------------------------------
    # 05  Predicted vs Ground Truth volume scatter
    # ----------------------------------------------------------
    if {"predicted_volume_cm3", "ground_truth_volume_cm3"} <= set(df.columns):
        pred = df["predicted_volume_cm3"].dropna()
        gt_v = df.loc[pred.index, "ground_truth_volume_cm3"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(gt_v, pred, color=color, alpha=0.6, s=40,
                   edgecolors="white", lw=0.4)
        all_v = pd.concat([gt_v, pred])
        lim   = (0, all_v.max() * 1.08)
        ax.plot(lim, lim, "k--", lw=1, label="Perfect agreement")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Ground Truth Volume (cm³)")
        ax.set_ylabel("Predicted Volume (cm³)")
        ax.set_title(f"{name}: Volume Agreement  (Predicted vs GT)")
        # linear regression line
        m_reg, b_reg, r_val, *_ = stats.linregress(gt_v, pred)
        x_line = np.linspace(*lim, 100)
        ax.plot(x_line, m_reg * x_line + b_reg, color="crimson", lw=1.3,
                label=f"Regression  r²={r_val**2:.3f}")
        ax.legend(fontsize=8)
        save_figure(fig, graph_dir, "05_volume_agreement_scatter.png")

    # ----------------------------------------------------------
    # 06  Signed relative volume error — histogram + KDE
    # ----------------------------------------------------------
    if "signed_relative_volume_error" in df.columns:
        vals = df["signed_relative_volume_error"].dropna().values
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(vals, bins=min(30, max(10, n // 3)),
                color=color, alpha=0.6, edgecolor="white", density=True)
        # KDE overlay
        from scipy.stats import gaussian_kde
        if len(vals) > 3:
            kde  = gaussian_kde(vals)
            xs   = np.linspace(vals.min() - 0.1, vals.max() + 0.1, 300)
            ax.plot(xs, kde(xs), color="black", lw=1.5, label="KDE")
        ax.axvline(0,              color="black",  lw=1.2, ls="-",  label="Zero error")
        ax.axvline(np.mean(vals),  color="crimson",lw=1.4, ls="--",
                   label=f"Mean = {np.mean(vals):+.3f}")
        ax.axvline(np.median(vals),color="darkblue",lw=1.2, ls=":",
                   label=f"Median = {np.median(vals):+.3f}")
        ax.set_xlabel("Signed Relative Volume Error  (pred - GT) / GT")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}: Signed Volume Error Distribution")
        ax.legend(fontsize=8)
        save_figure(fig, graph_dir, "06_signed_volume_error_distribution.png")

    # ----------------------------------------------------------
    # 07  Quality tier pie chart
    # ----------------------------------------------------------
    tiers_order  = ["Excellent", "Good", "Moderate", "Poor"]
    tier_col_map = {"Excellent": "#2ca02c", "Good": "#1f77b4",
                    "Moderate":  "#ff7f0e", "Poor": "#d62728"}
    tier_counts  = {t: int((df["quality_tier"] == t).sum()) for t in tiers_order}
    non_zero     = {t: c for t, c in tier_counts.items() if c > 0}
    if non_zero:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(list(non_zero.values()),
               labels=[f"{t}\n({c})" for t, c in non_zero.items()],
               colors=[tier_col_map[t] for t in non_zero],
               autopct="%1.1f%%", startangle=140,
               wedgeprops={"edgecolor": "white", "linewidth": 1.2})
        ax.set_title(f"{name}: Quality Tier Distribution  (n={n})")
        save_figure(fig, graph_dir, "07_quality_tier_pie.png")

    # ----------------------------------------------------------
    # 08  Failure mode horizontal bar chart
    # ----------------------------------------------------------
    fm_counts = df["failure_mode"].value_counts()
    if not fm_counts.empty:
        fig, ax = plt.subplots(figsize=(9, max(3, len(fm_counts) * 0.7 + 1)))
        bars = ax.barh(fm_counts.index[::-1], fm_counts.values[::-1],
                       color=color, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, fm_counts.values[::-1]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9)
        ax.set_xlabel("Number of Patients")
        ax.set_title(f"{name}: Failure Mode Breakdown")
        save_figure(fig, graph_dir, "08_failure_modes.png")

    # ----------------------------------------------------------
    # 09  Spearman correlation heatmap
    # ----------------------------------------------------------
    corr_cols = [c for c in CORRELATION_METRICS if c in df.columns]
    if len(corr_cols) >= 4:
        corr_mat = df[corr_cols].rank().corr()
        fig, ax  = plt.subplots(figsize=(max(8, len(corr_cols) * 0.75),
                                          max(7, len(corr_cols) * 0.75)))
        cmap = LinearSegmentedColormap.from_list(
            "rg", ["#d62728", "#ffffff", "#2ca02c"])
        im = ax.imshow(corr_mat.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
        tl = [c.replace("_", "\n") for c in corr_cols]
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(tl, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(tl, fontsize=7)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_mat.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(val) < 0.6 else "white")
        ax.set_title(f"{name}: Spearman Correlation Matrix")
        save_figure(fig, graph_dir, "09_correlation_heatmap.png")

    # ----------------------------------------------------------
    # 10  Pass-rate horizontal bar chart
    # ----------------------------------------------------------
    rates = pass_rates(df)
    if not rates.empty:
        fig, ax = plt.subplots(figsize=(9, max(4, len(rates) * 0.5 + 1.5)))
        ax.barh(rates["criterion"][::-1],
                rates["pass_rate_percent"][::-1],
                color=color, alpha=0.8, edgecolor="white")
        ax.axvline(50,  color="grey",   ls=":", lw=1)
        ax.axvline(100, color="black",  ls="-", lw=0.8)
        ax.set_xlim(0, 110)
        ax.set_xlabel("% Patients Passing")
        ax.set_title(f"{name}: Pass-Rate Summary (n={n})")
        for idx, row in rates.iterrows():
            ax.text(row["pass_rate_percent"] + 1,
                    len(rates) - 1 - idx,
                    f"{row['pass_rate_percent']:.1f}%  ({int(row['patients_passing'])}/{int(row['patients_evaluated'])})",
                    va="center", fontsize=8)
        save_figure(fig, graph_dir, "10_pass_rates.png")

    # ----------------------------------------------------------
    # 11  Sorted Dice with CI band
    # ----------------------------------------------------------
    if "dice" in df.columns:
        fig, ax = plt.subplots(figsize=(fw, 5))
        x    = np.arange(n)
        mean_d = _stat("dice", "mean")
        ci_lo  = _stat("dice", "ci95_low")
        ci_hi  = _stat("dice", "ci95_high")
        ax.plot(x, sorted_df["dice"], color=color, lw=1.5, marker=".", markersize=4)
        ax.axhline(mean_d, color="crimson", ls="--", lw=1.4,
                   label=f"Mean = {mean_d:.3f}")
        ax.axhspan(ci_lo, ci_hi, alpha=0.12, color="crimson",
                   label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        for t in [DICE_EXCELLENT, DICE_GOOD, DICE_MODERATE]:
            ax.axhline(t, color="#aaaaaa", ls=":", lw=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Dice Score")
        ax.set_title(f"{name}: Dice Score (sorted, with 95% CI band)")
        ax.legend(fontsize=8)
        _sparse_xticks(ax, slabels)
        save_figure(fig, graph_dir, "11_dice_sorted_ci_band.png")

    # ----------------------------------------------------------
    # 12  Volume error vs GT volume (bubble = abs error)
    # ----------------------------------------------------------
    if {"signed_relative_volume_error", "ground_truth_volume_cm3"} <= set(df.columns):
        sub = df[["signed_relative_volume_error",
                  "ground_truth_volume_cm3",
                  "absolute_volume_difference_mm3"]].dropna()
        if len(sub) > 2:
            sizes = (sub["absolute_volume_difference_mm3"] /
                     sub["absolute_volume_difference_mm3"].max() * 300 + 10)
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(sub["ground_truth_volume_cm3"],
                            sub["signed_relative_volume_error"],
                            s=sizes, c=sub["signed_relative_volume_error"],
                            cmap="RdYlGn_r", alpha=0.7,
                            edgecolors="white", lw=0.4)
            plt.colorbar(sc, ax=ax, label="Signed Relative Volume Error")
            ax.axhline(0, color="black", lw=1.2, ls="-")
            ax.set_xlabel("Ground Truth Volume (cm³)")
            ax.set_ylabel("Signed Relative Volume Error")
            ax.set_title(f"{name}: Volume Error vs Tumor Size")
            save_figure(fig, graph_dir, "12_volume_error_vs_gt_size.png")


# ============================================================
# EVALUATE INDIVIDUAL PIPELINE   (main per-pipeline entry)
# ============================================================

def evaluate_pipeline(pipeline_key: int, start: int, end: int):
    p_config   = PIPELINES[pipeline_key]
    range_name = f"{start:0{PATIENT_NUMBER_WIDTH}d}-{end:0{PATIENT_NUMBER_WIDTH}d}"
    out_dir    = os.path.join(p_config["stats_dir"], range_name)
    graph_dir  = os.path.join(out_dir, "Graphs")
    os.makedirs(graph_dir, exist_ok=True)

    print()
    print("=" * 75)
    print(f"ANALYZING: {p_config['name'].upper()}")
    print("=" * 75)

    records, missing, notes = load_range(start, end, p_config)
    for pid, note in missing:
        print(f"[SKIPPED] {pid}: {note}")
    if not records:
        print(f"No evaluation CSVs found for {p_config['name']} in this range.")
        return None, None

    df, inf_counts = build_patient_table(records)
    metrics        = numeric_metric_columns(df)

    summary  = summarize_metrics(df, metrics)
    pooled   = pooled_metrics(df)
    rates    = pass_rates(df)
    outliers = find_outliers(df)
    corr     = correlation_table(df)
    failures = failure_mode_table(df)
    tiers    = quality_tier_table(df)

    # ── CSV outputs ──────────────────────────────────────────
    front = [c for c in ["patient_id", "patient_number", "run_id", "region",
                          "quality_tier", "failure_mode"] if c in df.columns]
    table = df[front + metrics + ["source_file"]]
    table.to_csv(os.path.join(out_dir, "per_patient_metrics.csv"), index=False)

    ranking = table.sort_values("dice", ascending=False).copy()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    rank_cols = ["rank", "patient_id", "quality_tier", "failure_mode", "dice",
                 "iou_jaccard", "precision", "sensitivity_recall",
                 "specificity", "bounding_box_iou", "volume_similarity"] + [
        c for c in ["hd95_mm", "assd_mm", "signed_relative_volume_error"]
        if c in ranking.columns
    ]
    ranking[[c for c in rank_cols if c in ranking.columns]].to_csv(
        os.path.join(out_dir, "patient_ranking.csv"), index=False)

    summary.to_csv( os.path.join(out_dir, "summary_statistics.csv"),      index=False)
    pooled.to_csv(  os.path.join(out_dir, "pooled_metrics.csv"),           index=False)
    rates.to_csv(   os.path.join(out_dir, "threshold_pass_rates.csv"),     index=False)
    outliers.to_csv(os.path.join(out_dir, "outliers.csv"),                 index=False)
    failures.to_csv(os.path.join(out_dir, "failure_modes.csv"),            index=False)
    tiers.to_csv(   os.path.join(out_dir, "quality_tiers.csv"),            index=False)
    if not corr.empty:
        corr.to_csv(os.path.join(out_dir, "correlation_matrix_spearman.csv"))

    # ── Graphs ───────────────────────────────────────────────
    _plot_individual_pipeline(df, summary, p_config, graph_dir)

    # ── Conclusion text ──────────────────────────────────────
    S = summary.set_index("metric")
    def st(m, c):
        try: return float(S.loc[m, c])
        except: return np.nan

    with open(os.path.join(out_dir, "pipeline_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(f"""{'='*75}
 PIPELINE EVALUATION: {p_config['name'].upper()}
 Range: {make_patient_id(start)} -> {make_patient_id(end)}
{'='*75}
Patients evaluated      : {len(df)}

OVERLAP
  Mean Dice             : {fmt(st('dice','mean'),3)} +/- {fmt(st('dice','std'),3)}
                          Median {fmt(st('dice','median'),3)}  |  95% CI [{fmt(st('dice','ci95_low'),3)}, {fmt(st('dice','ci95_high'),3)}]
  Mean IoU (Jaccard)    : {fmt(st('iou_jaccard','mean'),3)}
  Mean Bounding-Box IoU : {fmt(st('bounding_box_iou','mean'),3)}
  Mean Volume Similarity: {fmt(st('volume_similarity','mean'),3)}

CLASSIFICATION
  Mean Sensitivity      : {fmt(st('sensitivity_recall','mean'),3)}
  Mean Precision        : {fmt(st('precision','mean'),3)}
  Mean Specificity      : {fmt(st('specificity','mean'),3)}
  Mean FPR              : {fmt(st('false_positive_rate','mean'),3)}
  Mean FNR              : {fmt(st('false_negative_rate','mean'),3)}
  Mean Accuracy         : {fmt(st('voxel_accuracy','mean'),3)}

BOUNDARY
  Mean HD95             : {fmt(st('hd95_mm','mean'),2)} mm
  Mean ASSD             : {fmt(st('assd_mm','mean'),2)} mm

VOLUME
  Mean Relative Error   : {fmt(st('relative_volume_error','mean'),3)}
  Mean Signed Rel Err   : {fmt(st('signed_relative_volume_error','mean'),3)}

QUALITY TIERS
  Excellent (>= 0.85)   : {int((df['quality_tier']=='Excellent').sum())} patients
  Good      (>= 0.70)   : {int((df['quality_tier']=='Good').sum())} patients
  Moderate  (>= 0.50)   : {int((df['quality_tier']=='Moderate').sum())} patients
  Poor      (<  0.50)   : {int((df['quality_tier']=='Poor').sum())} patients
{'='*75}
""")

    print(f"Saved {p_config['name']} statistics to: {out_dir}")
    return df, summary


# ============================================================
# TRI-PIPELINE COMPARATIVE ANALYSIS   (14 charts)
# ============================================================

def run_tri_pipeline_comparison(dfs: dict, summaries: dict, start: int, end: int):
    range_name = f"{start:0{PATIENT_NUMBER_WIDTH}d}-{end:0{PATIENT_NUMBER_WIDTH}d}"
    comp_dir   = os.path.join(COMPARATIVE_ROOT, range_name)
    graph_dir  = os.path.join(comp_dir, "Graphs")
    os.makedirs(graph_dir, exist_ok=True)

    print()
    print("=" * 80)
    print("   TRI-PIPELINE COMPARATIVE EVALUATION")
    print("   Thresholding  vs  Watershed  vs  Hybrid")
    print("=" * 80)
    print(f"Saving to: {comp_dir}")

    # ── columns to carry into the merge ─────────────────────
    MERGE_COLS = [
        "dice", "iou_jaccard", "precision", "sensitivity_recall",
        "specificity", "false_positive_rate", "false_negative_rate",
        "voxel_accuracy", "bounding_box_iou",
        "hd95_mm", "assd_mm",
        "relative_volume_error", "volume_similarity",
        "predicted_volume_cm3",
        "signed_relative_volume_error",
    ]

    def _prep(df, suffix, include_gt=False):
        keep = ["patient_id", "patient_number"] if include_gt else ["patient_id"]
        for c in MERGE_COLS:
            if c in df.columns:
                keep.append(c)
        if include_gt and "ground_truth_volume_cm3" in df.columns:
            keep.append("ground_truth_volume_cm3")
        sub = df[keep].copy()
        rename = {c: f"{c}_{suffix}"
                  for c in MERGE_COLS if c in sub.columns}
        return sub.rename(columns=rename)

    m1 = _prep(dfs[1], "p1", include_gt=True)
    m2 = _prep(dfs[2], "p2")
    m3 = _prep(dfs[3], "p3")

    merged = m1.merge(m2, on="patient_id").merge(m3, on="patient_id")
    n      = len(merged)

    # ── derived deltas ───────────────────────────────────────
    for m in ["dice", "iou_jaccard", "sensitivity_recall", "precision",
              "specificity", "volume_similarity", "hd95_mm", "assd_mm"]:
        c1, c2, c3 = f"{m}_p1", f"{m}_p2", f"{m}_p3"
        if {c1, c2, c3} <= set(merged.columns):
            merged[f"delta_{m}_p3_vs_p1"] = merged[c3] - merged[c1]
            merged[f"delta_{m}_p3_vs_p2"] = merged[c3] - merged[c2]
            merged[f"delta_{m}_p2_vs_p1"] = merged[c2] - merged[c1]

    # ── winner per patient ───────────────────────────────────
    def _winner(r):
        vals = {1: r.get("dice_p1", np.nan),
                2: r.get("dice_p2", np.nan),
                3: r.get("dice_p3", np.nan)}
        best = max(vals, key=lambda k: vals[k] if not pd.isna(vals[k]) else -1)
        return P_LABELS[best]

    merged["best_pipeline"] = merged.apply(_winner, axis=1)
    merged.to_csv(os.path.join(comp_dir, "comparative_patient_metrics_3way.csv"), index=False)

    # ── summary comparison table ─────────────────────────────
    s1 = summaries[1].set_index("metric")
    s2 = summaries[2].set_index("metric")
    s3 = summaries[3].set_index("metric")

    def _get(s, m, c):
        try: return float(s.loc[m, c])
        except: return np.nan

    comp_rows = []
    for m in COMPARATIVE_METRICS:
        # determine best (lower is better for error/distance metrics)
        lower_better = any(kw in m for kw in ["error", "hd95", "assd", "false"])
        means = {1: _get(s1, m, "mean"), 2: _get(s2, m, "mean"), 3: _get(s3, m, "mean")}
        if lower_better:
            best_key = min(means, key=lambda k: means[k] if not np.isnan(means[k]) else 1e9)
        else:
            best_key = max(means, key=lambda k: means[k] if not np.isnan(means[k]) else -1e9)
        comp_rows.append({
            "Metric":                   m,
            "P1_Mean":                  _get(s1, m, "mean"),
            "P1_Std":                   _get(s1, m, "std"),
            "P1_Median":                _get(s1, m, "median"),
            "P2_Mean":                  _get(s2, m, "mean"),
            "P2_Std":                   _get(s2, m, "std"),
            "P2_Median":                _get(s2, m, "median"),
            "P3_Mean":                  _get(s3, m, "mean"),
            "P3_Std":                   _get(s3, m, "std"),
            "P3_Median":                _get(s3, m, "median"),
            "Best_Pipeline":            P_LABELS[best_key],
            "P3_vs_P1_Delta_Mean":      means[3] - means[1],
            "P3_vs_P2_Delta_Mean":      means[3] - means[2],
            "P2_vs_P1_Delta_Mean":      means[2] - means[1],
        })
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(comp_dir, "tri_pipeline_summary_statistics.csv"), index=False)

    # ── win tally ─────────────────────────────────────────────
    tally = (merged["best_pipeline"].value_counts()
             .rename_axis("Pipeline").reset_index(name="Patient_Count"))
    tally["Win_Percentage"] = 100.0 * tally["Patient_Count"] / n
    tally.to_csv(os.path.join(comp_dir, "tri_pipeline_win_tally.csv"), index=False)

    # ── all pairwise significance tests + effect size ─────────
    test_pairs = [(3, 1, "P3_vs_P1"), (3, 2, "P3_vs_P2"), (2, 1, "P2_vs_P1")]
    sig_rows = []
    for m in ["dice", "iou_jaccard", "sensitivity_recall", "precision",
              "specificity", "bounding_box_iou", "volume_similarity",
              "hd95_mm", "assd_mm", "relative_volume_error"]:
        row = {"Metric": m}
        for (ka, kb, label) in test_pairs:
            ca, cb = f"{m}_p{ka}", f"{m}_p{kb}"
            if ca not in merged.columns or cb not in merged.columns:
                continue
            va = merged[ca].dropna()
            vb = merged[cb].dropna()
            # align indices
            common = va.index.intersection(vb.index)
            va, vb = va.loc[common].values, vb.loc[common].values
            if len(va) < 5:
                continue
            try:
                _, p_t  = stats.ttest_rel(va, vb)
                _, p_w  = stats.wilcoxon(va, vb)
            except Exception:
                p_t = p_w = np.nan
            d = cohens_d(va, vb)
            row.update({
                f"{label}_Mean_Delta": float((va - vb).mean()),
                f"{label}_t_pval":     float(p_t),
                f"{label}_Wilcoxon_pval": float(p_w),
                f"{label}_Sig":        bool((p_t < 0.05) or (p_w < 0.05)),
                f"{label}_CohenD":     float(d),
            })
        sig_rows.append(row)
    pd.DataFrame(sig_rows).to_csv(
        os.path.join(comp_dir, "tri_pipeline_significance_tests.csv"), index=False)

    # ── quality tier comparison table ─────────────────────────
    tier_rows = []
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        tier_rows.append({
            "Quality_Tier": tier,
            "P1_Count":     int((dfs[1]["quality_tier"] == tier).sum()),
            "P2_Count":     int((dfs[2]["quality_tier"] == tier).sum()),
            "P3_Count":     int((dfs[3]["quality_tier"] == tier).sum()),
            "P1_Pct":       100 * (dfs[1]["quality_tier"] == tier).mean(),
            "P2_Pct":       100 * (dfs[2]["quality_tier"] == tier).mean(),
            "P3_Pct":       100 * (dfs[3]["quality_tier"] == tier).mean(),
        })
    pd.DataFrame(tier_rows).to_csv(
        os.path.join(comp_dir, "tri_pipeline_quality_tiers.csv"), index=False)

    # ── failure mode cross-table ───────────────────────────────
    fm_rows = []
    all_modes = set(dfs[1]["failure_mode"].unique()) | \
                set(dfs[2]["failure_mode"].unique()) | \
                set(dfs[3]["failure_mode"].unique())
    for mode in sorted(all_modes):
        fm_rows.append({
            "Failure_Mode": mode,
            "P1_Count": int((dfs[1]["failure_mode"] == mode).sum()),
            "P2_Count": int((dfs[2]["failure_mode"] == mode).sum()),
            "P3_Count": int((dfs[3]["failure_mode"] == mode).sum()),
        })
    pd.DataFrame(fm_rows).to_csv(
        os.path.join(comp_dir, "tri_pipeline_failure_modes.csv"), index=False)

    # ==========================================================
    # COMPARATIVE VISUALIZATIONS
    # ==========================================================
    fw     = _fig_width(n)
    x      = np.arange(n)
    # sort merged by P3 dice for all per-patient plots
    merged_s = merged.sort_values("dice_p3", ascending=False).reset_index(drop=True)
    labels_s = merged_s["patient_number"].tolist()

    # ── 01  Sorted dice line plot (most informative at n=100) ─
    if {"dice_p1", "dice_p2", "dice_p3"} <= set(merged_s.columns):
        fig, ax = plt.subplots(figsize=(fw, 5))
        for key, col_sfx, mk in [(1, "p1", "o"), (2, "p2", "^"), (3, "p3", "s")]:
            col = f"dice_{col_sfx}"
            ax.plot(x, merged_s[col], color=P_COLORS[key], lw=1.4,
                    label=f"{P_LABELS[key]}  μ={merged_s[col].mean():.3f}",
                    marker=mk, markersize=3, alpha=0.85)
        for t in [DICE_EXCELLENT, DICE_GOOD, DICE_MODERATE]:
            ax.axhline(t, color="#bbbbbb", ls=":", lw=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Dice Score")
        ax.set_title("All 3 Pipelines: Dice Score per Patient (sorted by P3 Hybrid)")
        ax.legend(fontsize=9)
        _sparse_xticks(ax, labels_s)
        save_figure(fig, graph_dir, "01_comparative_dice_sorted_lines.png")

    # ── 02  Grouped boxplots for all COMPARATIVE_METRICS ──────
    avail_m = [m for m in COMPARATIVE_METRICS
               if f"{m}_p1" in merged.columns and
                  f"{m}_p2" in merged.columns and
                  f"{m}_p3" in merged.columns]
    if avail_m:
        fig, axes = plt.subplots(1, len(avail_m),
                                  figsize=(max(14, len(avail_m) * 2.0), 7),
                                  sharey=False)
        if len(avail_m) == 1:
            axes = [axes]
        for ax, m in zip(axes, avail_m):
            data = [merged[f"{m}_p1"].dropna().values,
                    merged[f"{m}_p2"].dropna().values,
                    merged[f"{m}_p3"].dropna().values]
            bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops={"color": "black", "lw": 2})
            for patch, key in zip(bp["boxes"], [1, 2, 3]):
                patch.set_facecolor(P_COLORS[key])
                patch.set_alpha(0.7)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["P1", "P2", "P3"], fontsize=8)
            ax.set_title(m.replace("_", "\n"), fontsize=8)
        legend_patches = [mpatches.Patch(color=P_COLORS[k], label=P_LABELS[k])
                          for k in [1, 2, 3]]
        fig.legend(handles=legend_patches, loc="upper right", fontsize=9)
        fig.suptitle("All Metrics: Pipeline Distribution Comparison", fontsize=11)
        save_figure(fig, graph_dir, "02_all_metrics_boxplots.png")

    # ── 03  Precision-Recall scatter (all 3 pipelines) ────────
    if all(c in merged.columns for c in
           ["sensitivity_recall_p1", "precision_p1",
            "sensitivity_recall_p2", "precision_p2",
            "sensitivity_recall_p3", "precision_p3"]):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        for key, sfx, mk in [(1, "p1", "o"), (2, "p2", "^"), (3, "p3", "s")]:
            ax.scatter(merged[f"sensitivity_recall_{sfx}"],
                       merged[f"precision_{sfx}"],
                       color=P_COLORS[key], marker=mk, s=45, alpha=0.65,
                       edgecolors="white", lw=0.4, label=P_LABELS[key])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Recall — All 3 Pipelines")
        ax.legend(fontsize=9)
        save_figure(fig, graph_dir, "03_precision_recall_scatter.png")

    # ── 04  Delta Dice: P3 vs P1 and P3 vs P2 (dual panel) ───
    for idx, (da, db, title, fname) in enumerate([
        ("delta_dice_p3_vs_p1", "delta_dice_p2_vs_p1",
         "Dice Delta: Hybrid vs Thresholding  |  Watershed vs Thresholding",
         "04_delta_dice_p3_vs_p1_and_p2_vs_p1.png"),
        ("delta_dice_p3_vs_p2", "delta_dice_p2_vs_p1",
         "Dice Delta: Hybrid vs Watershed  |  Watershed vs Thresholding",
         "05_delta_dice_p3_vs_p2_and_p2_vs_p1.png"),
    ]):
        if da not in merged_s.columns:
            continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fw, 8), sharex=True)
        for ax, col, main_color, zero_label in [
            (ax1, da, P_COLORS[3], "Hybrid"),
            (ax2, db if db in merged_s.columns else da,
             P_COLORS[2], "Watershed"),
        ]:
            vals   = merged_s[col]
            colors = [main_color if v >= 0 else "#d62728" for v in vals]
            ax.bar(x, vals, color=colors, edgecolor="white", lw=0.3, width=0.8)
            ax.axhline(0, color="black", lw=1)
            ax.axhline(vals.mean(), color="purple", ls="--", lw=1.3,
                       label=f"Mean {vals.mean():+.3f}")
            ax.set_ylabel("ΔDice")
            ax.legend(fontsize=8)
        ax2.set_xlabel("Patient (sorted by P3 Dice)")
        fig.suptitle(title, fontsize=10)
        _sparse_xticks(ax2, labels_s)
        save_figure(fig, graph_dir, fname)

    # ── 06  HD95 & ASSD violin comparison ─────────────────────
    for m, fn in [("hd95_mm", "06_hd95_violin_comparison.png"),
                  ("assd_mm", "07_assd_violin_comparison.png")]:
        cols = [f"{m}_p1", f"{m}_p2", f"{m}_p3"]
        if not all(c in merged.columns for c in cols):
            continue
        data = [merged[c].dropna().values for c in cols]
        fig, ax = plt.subplots(figsize=(7, 6))
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc, key in zip(parts["bodies"], [1, 2, 3]):
            pc.set_facecolor(P_COLORS[key])
            pc.set_alpha(0.65)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([P_LABELS[k] for k in [1, 2, 3]], fontsize=9)
        ax.set_ylabel("mm")
        ax.set_title(f"{m.upper()} Distribution — All 3 Pipelines")
        means = [np.nanmean(d) for d in data]
        for i, (val, key) in enumerate(zip(means, [1, 2, 3])):
            ax.text(i + 1, val, f"μ={val:.1f}", ha="center", va="bottom",
                    fontsize=8, color=P_COLORS[key], fontweight="bold")
        save_figure(fig, graph_dir, fn)

    # ── 08  Volume agreement scatter (all 3 pipelines) ────────
    gt_col = "ground_truth_volume_cm3"
    if gt_col in merged.columns:
        pred_cols = [c for c in [f"predicted_volume_cm3_p{k}" for k in [1, 2, 3]]
                     if c in merged.columns]
        if pred_cols:
            fig, ax = plt.subplots(figsize=(7, 7))
            all_vals = merged[gt_col].values
            for key, sfx, mk in [(1, "p1", "o"), (2, "p2", "^"), (3, "p3", "s")]:
                col = f"predicted_volume_cm3_{sfx}"
                if col not in merged.columns:
                    continue
                ax.scatter(merged[gt_col], merged[col],
                           color=P_COLORS[key], marker=mk, s=45, alpha=0.6,
                           edgecolors="white", lw=0.4, label=P_LABELS[key])
            lim = (0, merged[gt_col].max() * 1.1)
            ax.plot(lim, lim, "k--", lw=1, label="Identity")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("Ground Truth Volume (cm³)")
            ax.set_ylabel("Predicted Volume (cm³)")
            ax.set_title("Volume Agreement — All 3 Pipelines vs Ground Truth")
            ax.legend(fontsize=9)
            save_figure(fig, graph_dir, "08_volume_agreement_all_pipelines.png")

    # ── 09  Win tally pie chart ────────────────────────────────
    tally_dict = merged["best_pipeline"].value_counts().to_dict()
    if tally_dict:
        fig, ax = plt.subplots(figsize=(6, 6))
        tally_colors = [P_COLORS[k] for k in [1, 2, 3]
                        if P_LABELS[k] in tally_dict]
        tally_labels = [k for k in [P_LABELS[1], P_LABELS[2], P_LABELS[3]]
                        if k in tally_dict]
        tally_vals   = [tally_dict[k] for k in tally_labels]
        ax.pie(tally_vals,
               labels=[f"{l}\n({v})" for l, v in zip(tally_labels, tally_vals)],
               colors=[P_COLORS[k] for k in [1, 2, 3] if P_LABELS[k] in tally_dict],
               autopct="%1.1f%%", startangle=140,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title(f"Best Pipeline per Patient  (n={n}, ranked by Dice)")
        save_figure(fig, graph_dir, "09_win_tally_pie.png")

    # ── 10  Radar chart of mean metrics ───────────────────────
    radar_metrics = [m for m in
                     ["dice", "iou_jaccard", "precision", "sensitivity_recall",
                      "specificity", "bounding_box_iou", "volume_similarity"]
                     if f"{m}_p1" in merged.columns]
    if len(radar_metrics) >= 4:
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7),
                               subplot_kw={"projection": "polar"})
        for key, sfx in [(1, "p1"), (2, "p2"), (3, "p3")]:
            vals = [merged[f"{m}_{sfx}"].mean() for m in radar_metrics]
            vals += vals[:1]
            ax.plot(angles, vals, color=P_COLORS[key], lw=2,
                    label=f"{P_LABELS[key]}  Dice μ={merged[f'dice_{sfx}'].mean():.3f}")
            ax.fill(angles, vals, color=P_COLORS[key], alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", "\n") for m in radar_metrics], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_title("Mean Metric Radar — All 3 Pipelines", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        save_figure(fig, graph_dir, "10_radar_mean_metrics.png")

    # ── 11  Signed volume error violin (over/under seg) ───────
    sve_cols = [f"signed_relative_volume_error_p{k}" for k in [1, 2, 3]]
    if all(c in merged.columns for c in sve_cols):
        data = [merged[c].dropna().values for c in sve_cols]
        fig, ax = plt.subplots(figsize=(7, 6))
        parts = ax.violinplot(data, showmedians=True)
        for pc, key in zip(parts["bodies"], [1, 2, 3]):
            pc.set_facecolor(P_COLORS[key])
            pc.set_alpha(0.65)
        ax.axhline(0, color="black", lw=1.3, ls="-", label="Zero error")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([P_LABELS[k] for k in [1, 2, 3]], fontsize=9)
        ax.set_ylabel("(Predicted - GT) / GT")
        ax.set_title("Signed Relative Volume Error — Over/Under-Segmentation")
        ax.legend(fontsize=8)
        save_figure(fig, graph_dir, "11_signed_volume_error_violin.png")

    # ── 12  Cohen's d effect size bar chart ───────────────────
    sig_df  = pd.DataFrame(sig_rows)
    d_pairs = [("P3_vs_P1", P_COLORS[3], "P3 vs P1"),
               ("P3_vs_P2", P_COLORS[2], "P3 vs P2"),
               ("P2_vs_P1", "#9467bd",   "P2 vs P1")]
    d_cols  = [c for _, _, _ in d_pairs
               if f"{_[0]}_CohenD" in sig_df.columns
               for _ in [(_,)]]

    effect_cols = [f"{label}_CohenD" for _, _, label in
                   [("P3_vs_P1", P_COLORS[3], "P3_vs_P1"),
                    ("P3_vs_P2", P_COLORS[2], "P3_vs_P2"),
                    ("P2_vs_P1", "#9467bd",   "P2_vs_P1")]]
    avail_effect = [c for c in effect_cols if c in sig_df.columns]

    if not sig_df.empty and avail_effect and "Metric" in sig_df.columns:
        sub_df = sig_df[["Metric"] + avail_effect].dropna(subset=avail_effect, how="all")
        bw = 0.25
        xe = np.arange(len(sub_df))
        fig, ax = plt.subplots(figsize=(max(10, len(sub_df) * 0.9 + 2), 6))
        for i, (col, lbl, clr) in enumerate(
            zip(avail_effect,
                ["P3 vs P1", "P3 vs P2", "P2 vs P1"],
                [P_COLORS[3], P_COLORS[2], "#9467bd"])):
            ax.bar(xe + (i - 1) * bw, sub_df[col].fillna(0),
                   bw, label=lbl, color=clr, alpha=0.8, edgecolor="white")
        ax.axhline(0, color="black", lw=1)
        ax.axhline( 0.2, color="#aaaaaa", ls=":", lw=0.9, label="Small (0.2)")
        ax.axhline( 0.5, color="#888888", ls=":", lw=0.9, label="Medium (0.5)")
        ax.axhline( 0.8, color="#555555", ls=":", lw=0.9, label="Large (0.8)")
        ax.axhline(-0.2, color="#aaaaaa", ls=":", lw=0.9)
        ax.axhline(-0.5, color="#888888", ls=":", lw=0.9)
        ax.set_xticks(xe)
        ax.set_xticklabels(sub_df["Metric"].tolist(), rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Cohen's d")
        ax.set_title("Effect Size (Cohen's d) — All Pairwise Pipeline Comparisons")
        ax.legend(fontsize=9)
        save_figure(fig, graph_dir, "12_effect_size_cohens_d.png")

    # ── 13  Significance heatmap (p-values) ───────────────────
    if not sig_df.empty and "Metric" in sig_df.columns:
        p_cols = [c for c in sig_df.columns if "pval" in c.lower()]
        if p_cols:
            pmat = sig_df[["Metric"] + p_cols].set_index("Metric").apply(
                pd.to_numeric, errors="coerce")
            fig, ax = plt.subplots(figsize=(max(6, len(p_cols) * 1.6),
                                            max(5, len(pmat) * 0.55 + 1.5)))
            cmap_p = LinearSegmentedColormap.from_list(
                "pmap", ["#d62728", "#ff7f0e", "#ffffff"])
            im = ax.imshow(pmat.values, cmap=cmap_p, vmin=0, vmax=0.1,
                           aspect="auto")
            plt.colorbar(im, ax=ax, label="p-value  (< 0.05 = significant)")
            ax.set_xticks(range(len(p_cols)))
            ax.set_xticklabels([c.replace("_pval", "").replace("_", "\n")
                                 for c in p_cols], fontsize=8)
            ax.set_yticks(range(len(pmat)))
            ax.set_yticklabels(pmat.index.tolist(), fontsize=8)
            for i in range(len(pmat)):
                for j, col in enumerate(p_cols):
                    val = pmat.iloc[i, j]
                    marker = "✓" if (not np.isnan(val) and val < 0.05) else "·"
                    ax.text(j, i, f"{val:.3f}\n{marker}",
                            ha="center", va="center", fontsize=7)
            ax.set_title("Statistical Significance Heatmap  (p-values, α=0.05)")
            save_figure(fig, graph_dir, "13_significance_heatmap.png")

    # ── 14  Cumulative Dice distribution (CDF) ────────────────
    if {"dice_p1", "dice_p2", "dice_p3"} <= set(merged.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        for key, sfx in [(1, "p1"), (2, "p2"), (3, "p3")]:
            d = np.sort(merged[f"dice_{sfx}"].dropna().values)
            cdf = np.arange(1, len(d) + 1) / len(d)
            ax.plot(d, cdf, color=P_COLORS[key], lw=2, label=P_LABELS[key])
        for t in [DICE_MODERATE, DICE_GOOD, DICE_EXCELLENT]:
            ax.axvline(t, color="#aaaaaa", ls=":", lw=0.9)
            ax.text(t + 0.005, 0.02, f"{t}", fontsize=7, color="#888888")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Cumulative Fraction of Patients")
        ax.set_title("Cumulative Dice Distribution (CDF) — All 3 Pipelines")
        ax.legend(fontsize=9)
        save_figure(fig, graph_dir, "14_cumulative_dice_cdf.png")

    # ── 15  Quality tier stacked bar ──────────────────────────
    tier_order  = ["Excellent", "Good", "Moderate", "Poor"]
    tier_colors_map = {"Excellent": "#2ca02c", "Good": "#1f77b4",
                       "Moderate": "#ff7f0e", "Poor": "#d62728"}
    tier_data = {t: [int((dfs[k]["quality_tier"] == t).sum()) for k in [1, 2, 3]]
                 for t in tier_order}
    fig, ax = plt.subplots(figsize=(7, 6))
    bottoms = [0, 0, 0]
    for tier in tier_order:
        vals = tier_data[tier]
        ax.bar([P_LABELS[k] for k in [1, 2, 3]], vals,
               bottom=bottoms, color=tier_colors_map[tier],
               edgecolor="white", lw=0.5, label=tier)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Number of Patients")
    ax.set_title("Quality Tier Distribution — All 3 Pipelines")
    ax.legend(loc="upper right", fontsize=9)
    save_figure(fig, graph_dir, "15_quality_tier_stacked_bar.png")

    # ── 16  Per-metric improvement heatmap ────────────────────
    # Shows mean delta for every metric × every pipeline pair
    pairs_lbl = ["P3-P1", "P3-P2", "P2-P1"]
    delta_cols_map = {
        "P3-P1": "delta_{m}_p3_vs_p1",
        "P3-P2": "delta_{m}_p3_vs_p2",
        "P2-P1": "delta_{m}_p2_vs_p1",
    }
    heat_metrics = [m for m in COMPARATIVE_METRICS if any(
        delta_cols_map[p].format(m=m) in merged.columns for p in pairs_lbl)]
    if heat_metrics:
        heat_data = np.full((len(heat_metrics), len(pairs_lbl)), np.nan)
        for i, m in enumerate(heat_metrics):
            for j, p in enumerate(pairs_lbl):
                col = delta_cols_map[p].format(m=m)
                if col in merged.columns:
                    heat_data[i, j] = merged[col].mean()
        fig, ax = plt.subplots(figsize=(6, max(5, len(heat_metrics) * 0.55 + 2)))
        abs_max = np.nanmax(np.abs(heat_data)) + 1e-9
        cmap_d  = LinearSegmentedColormap.from_list(
            "div", ["#d62728", "#ffffff", "#2ca02c"])
        im = ax.imshow(heat_data, cmap=cmap_d,
                       vmin=-abs_max, vmax=abs_max, aspect="auto")
        plt.colorbar(im, ax=ax, label="Mean Δ (positive = row pipeline better)")
        ax.set_xticks(range(len(pairs_lbl)))
        ax.set_xticklabels(pairs_lbl, fontsize=10)
        ax.set_yticks(range(len(heat_metrics)))
        ax.set_yticklabels(heat_metrics, fontsize=8)
        for i in range(len(heat_metrics)):
            for j in range(len(pairs_lbl)):
                val = heat_data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if abs(val) < abs_max * 0.6 else "white")
        ax.set_title("Mean Per-Metric Improvement Heatmap\n(All Pairwise Pipeline Comparisons)")
        save_figure(fig, graph_dir, "16_improvement_heatmap.png")

    # ── Conclusion text ────────────────────────────────────────
    def _mu(col):
        return f"{merged[col].mean():.4f}" if col in merged.columns else "n/a"

    with open(os.path.join(comp_dir, "tri_pipeline_comparison_conclusion.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"""{'='*78}
 TRI-PIPELINE COMPARATIVE EVALUATION
 Range: {make_patient_id(start)} -> {make_patient_id(end)}
 Patients compared: {n}
{'='*78}

MEAN DICE SCORES
  Pipeline 1 (Thresholding) : {_mu('dice_p1')}
  Pipeline 2 (Watershed)    : {_mu('dice_p2')}
  Pipeline 3 (Hybrid)       : {_mu('dice_p3')}

MEAN IoU (JACCARD)
  Pipeline 1                : {_mu('iou_jaccard_p1')}
  Pipeline 2                : {_mu('iou_jaccard_p2')}
  Pipeline 3                : {_mu('iou_jaccard_p3')}

MEAN HD95 (mm)
  Pipeline 1                : {_mu('hd95_mm_p1')}
  Pipeline 2                : {_mu('hd95_mm_p2')}
  Pipeline 3                : {_mu('hd95_mm_p3')}

MEAN SENSITIVITY (RECALL)
  Pipeline 1                : {_mu('sensitivity_recall_p1')}
  Pipeline 2                : {_mu('sensitivity_recall_p2')}
  Pipeline 3                : {_mu('sensitivity_recall_p3')}

MEAN PRECISION
  Pipeline 1                : {_mu('precision_p1')}
  Pipeline 2                : {_mu('precision_p2')}
  Pipeline 3                : {_mu('precision_p3')}

MEAN VOLUME SIMILARITY
  Pipeline 1                : {_mu('volume_similarity_p1')}
  Pipeline 2                : {_mu('volume_similarity_p2')}
  Pipeline 3                : {_mu('volume_similarity_p3')}

WIN TALLY (best Dice per patient)
{tally.to_string(index=False)}

DELTA DICE: P3 (Hybrid) vs P1 (Thresholding)
  Mean  : {_mu('delta_dice_p3_vs_p1')}
DELTA DICE: P3 (Hybrid) vs P2 (Watershed)
  Mean  : {_mu('delta_dice_p3_vs_p2')}
DELTA DICE: P2 (Watershed) vs P1 (Thresholding)
  Mean  : {_mu('delta_dice_p2_vs_p1')}
{'='*78}
""")

    print(f"Comparative analysis complete. Outputs saved to: {comp_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 75)
    print("  BRAIN TUMOR SEGMENTATION — RANGE STATISTICS & COMPARATIVE ANALYSIS")
    print("=" * 75)
    print()
    print("Select mode:")
    print("  [1]  Pipeline 1.0  (Thresholding)  vs Ground Truth")
    print("  [2]  Pipeline 2.0  (Watershed)      vs Ground Truth")
    print("  [3]  Pipeline 3.0  (Hybrid)          vs Ground Truth")
    print("  [4]  All 3 pipelines  +  Full Tri-Pipeline Comparative Analysis")
    print()

    while True:
        try:
            mode = int(input("Mode (1/2/3/4): ").strip())
            if mode in [1, 2, 3, 4]:
                break
        except ValueError:
            pass
        print("  Please enter 1, 2, 3, or 4.")

    print()
    print(f"Patient ID format: {make_patient_id(1)} ... {make_patient_id(100)}")
    start = ask_integer("Start patient number : ")
    while True:
        end = ask_integer("End patient number   : ")
        if end >= start:
            break
        print("  End must be >= start.")

    if mode in [1, 2, 3]:
        evaluate_pipeline(mode, start, end)
    elif mode == 4:
        dfs, summaries = {}, {}
        for p_key in [1, 2, 3]:
            df, summary = evaluate_pipeline(p_key, start, end)
            if df is not None:
                dfs[p_key]       = df
                summaries[p_key] = summary

        if len(dfs) == 3:
            run_tri_pipeline_comparison(dfs, summaries, start, end)
        else:
            print("\nWarning: Could not run tri-pipeline comparison — "
                  "some pipelines lacked evaluation data.")


if __name__ == "__main__":
    main()
