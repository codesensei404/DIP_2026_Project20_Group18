"""
Range statistics for the brain tumor segmentation pipeline.

Reads ONLY the segmentation-evaluation CSV of every patient found in
    Results/Tables/<patient_id>/
(the morphological-analysis CSV is ignored automatically) and writes
every table, graph and the overall conclusion to
    Results/Range_Statistics/<start>-<end>/      e.g. 001-020

Place this file in the Codes/ folder (next to the other modules).
It finds the project root itself, so it can be run from anywhere:
    python Codes/range_statistics.py            # asks for the range
    python Codes/range_statistics.py 1 20       # or give it directly
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PATIENT_PREFIX = "BraTS20_Training_"
PATIENT_NUMBER_WIDTH = 3

# Same region that main.py evaluates (files are named
# <patient_id>_<REGION>_evaluation.csv, e.g. BraTS20_Training_001_WT_evaluation.csv)
EVALUATION_REGION = "WT"

# This file lives in <project>/Codes/, so the project root is one level up.
# (If the file is placed directly in the root instead, that also works.)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = (
    os.path.dirname(_THIS_DIR)
    if os.path.basename(_THIS_DIR).lower() == "codes"
    else _THIS_DIR
)
TABLES_ROOT = os.path.join(PROJECT_DIR, "Results", "Tables")
STATS_ROOT = os.path.join(PROJECT_DIR, "Results", "Range_Statistics")

# Dice quality tiers
DICE_EXCELLENT = 0.85
DICE_GOOD = 0.70
DICE_MODERATE = 0.50

# Precision/recall gap needed to call a failure "over" or "under" segmentation
FAILURE_MARGIN = 0.10

# Thresholds used for pass-rate tables
DICE_THRESHOLDS = [0.50, 0.70, 0.80, 0.85, 0.90]
HD95_THRESHOLDS_MM = [5, 10, 20]

BOOTSTRAP_SAMPLES = 2000

# Metric groups (controls the order and the "group" column of the summary)
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
    "hd95_mm", "assd_mm", "relative_volume_error",
]

CORRELATION_METRICS = [
    "dice", "iou_jaccard", "precision", "sensitivity_recall",
    "specificity", "hd95_mm", "assd_mm", "ground_truth_volume_mm3",
    "predicted_volume_mm3", "relative_volume_error", "volume_similarity",
]


# ============================================================
# HELPERS
# ============================================================

def make_patient_id(number):
    return f"{PATIENT_PREFIX}{number:0{PATIENT_NUMBER_WIDTH}d}"


def ask_integer(prompt):
    while True:
        value = input(prompt).strip()
        try:
            number = int(value)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if number < 1:
            print("  Please enter a number of 1 or more.")
            continue
        return number


def get_range():
    if len(sys.argv) >= 3:
        try:
            start, end = int(sys.argv[1]), int(sys.argv[2])
            if 1 <= start <= end:
                return start, end
        except ValueError:
            pass
        print("Invalid command-line range, asking instead.")

    print()
    print("=" * 75)
    print("        BRAIN TUMOR PIPELINE - RANGE STATISTICS")
    print("=" * 75)
    print()
    print("Enter the range of patients to analyse.")
    print(f"Example: 1 to 20 -> {make_patient_id(1)} ... {make_patient_id(20)}")
    print()

    start = ask_integer("Start patient number : ")
    while True:
        end = ask_integer("End patient number   : ")
        if end >= start:
            return start, end
        print("  End must be greater than or equal to start.")


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


# ============================================================
# LOADING THE SEGMENTATION-EVALUATION CSVs
# ============================================================

def read_metric_csv(path):
    """
    Reads a 'Metric , Value' CSV and returns {metric: value}.
    Returns None if the file is not a segmentation-evaluation table
    (identified by the presence of the 'dice' metric), so the
    morphological-analysis CSV is skipped automatically.
    """
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


def find_evaluation_csv(patient_id):
    """
    Finds the segmentation-evaluation CSV inside
    Results/Tables/<patient_id>/.

    1. Preferred: <patient_id>_<REGION>_evaluation.csv
       (e.g. BraTS20_Training_001_WT_evaluation.csv)
    2. Fallback: any other CSV whose content is an evaluation table
       (it has a 'dice' row) and whose region matches EVALUATION_REGION.
    The *_morphological_analysis.csv and all .json files are ignored.
    """
    table_dir = os.path.join(TABLES_ROOT, patient_id)

    if not os.path.isdir(table_dir):
        return None, None, "table folder not found"

    expected = os.path.join(
        table_dir,
        f"{patient_id}_{EVALUATION_REGION}_evaluation.csv"
    )

    if os.path.isfile(expected):
        data = read_metric_csv(expected)
        if data is not None:
            return expected, data, ""

    csv_files = [
        os.path.join(table_dir, name)
        for name in os.listdir(table_dir)
        if name.lower().endswith(".csv")
        and "morphological" not in name.lower()
    ]

    csv_files.sort(key=os.path.getmtime, reverse=True)

    valid = []
    for path in csv_files:
        data = read_metric_csv(path)
        if data is None:
            continue
        if data.get("region", EVALUATION_REGION) != EVALUATION_REGION:
            continue
        valid.append((path, data))

    if not valid:
        return None, None, (
            f"no {EVALUATION_REGION} segmentation-evaluation CSV found"
        )

    note = (
        f"expected file not found, used {os.path.basename(valid[0][0])}"
    )

    return valid[0][0], valid[0][1], note


def load_range(start, end):
    records = []
    missing = []
    notes = []

    for number in range(start, end + 1):
        patient_id = make_patient_id(number)
        path, data, note = find_evaluation_csv(patient_id)

        if data is None:
            missing.append((patient_id, note))
            continue

        if note:
            notes.append((patient_id, note))

        row = dict(data)
        row["patient_id"] = row.get("patient_id", patient_id)
        row["source_file"] = os.path.basename(path)
        records.append(row)

    return records, missing, notes


# ============================================================
# BUILDING THE PER-PATIENT TABLE
# ============================================================

def build_patient_table(records):
    df = pd.DataFrame(records)

    text_columns = IDENTITY_COLUMNS + ["source_file"]
    inf_counts = {}

    for column in df.columns:
        if column in text_columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        infinite = int(np.isinf(values).sum())
        if infinite:
            inf_counts[column] = infinite
        df[column] = values.replace([np.inf, -np.inf], np.nan)

    df["patient_number"] = (
        df["patient_id"].str.split("_").str[-1]
    )

    # ---- derived (signed) volume measures -------------------
    if {"predicted_volume_mm3", "ground_truth_volume_mm3"} <= set(df.columns):
        df["signed_volume_difference_mm3"] = (
            df["predicted_volume_mm3"] - df["ground_truth_volume_mm3"]
        )
        gt = df["ground_truth_volume_mm3"].replace(0, np.nan)
        df["signed_relative_volume_error"] = (
            df["signed_volume_difference_mm3"] / gt
        )

    df["quality_tier"] = df["dice"].apply(quality_tier)
    df["failure_mode"] = df.apply(failure_mode, axis=1)

    return df, inf_counts


def quality_tier(dice):
    if pd.isna(dice):
        return "No data"
    if dice >= DICE_EXCELLENT:
        return "Excellent"
    if dice >= DICE_GOOD:
        return "Good"
    if dice >= DICE_MODERATE:
        return "Moderate"
    return "Poor"


def failure_mode(row):
    dice = row.get("dice")
    precision = row.get("precision")
    recall = row.get("sensitivity_recall")

    if pd.isna(dice):
        return "No data"
    if dice >= DICE_EXCELLENT:
        return "None (excellent overlap)"
    if pd.isna(precision) or pd.isna(recall):
        return "Unknown"
    if recall - precision > FAILURE_MARGIN:
        return "Over-segmentation (extra false positives)"
    if precision - recall > FAILURE_MARGIN:
        return "Under-segmentation (missed tumor)"
    return "Mixed (boundary errors)"


def numeric_metric_columns(df):
    skip = set(IDENTITY_COLUMNS + [
        "source_file", "patient_number", "quality_tier", "failure_mode",
    ])
    columns = [c for c in df.columns if c not in skip]

    ordered = []
    for names in GROUPS.values():
        ordered += [n for n in names if n in columns]
    ordered += [c for c in columns if c not in ordered]
    return ordered


def group_of(metric):
    for group, names in GROUPS.items():
        if metric in names:
            return group
    return "Other"


# ============================================================
# STATISTICS
# ============================================================

def bootstrap_ci(values, rng):
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, n), replace=True)
    means = samples.mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarize_metrics(df, metrics):
    rng = np.random.default_rng(0)
    rows = []

    for metric in metrics:
        series = df[metric]
        x = series.dropna().to_numpy(dtype=float)
        n = len(x)

        row = {
            "metric": metric,
            "group": group_of(metric),
            "n_valid": n,
            "n_missing": int(series.isna().sum()),
        }

        if n == 0:
            rows.append(row)
            continue

        mean = x.mean()
        std = x.std(ddof=1) if n > 1 else np.nan
        q1, median, q3 = np.percentile(x, [25, 50, 75])
        ci_low, ci_high = bootstrap_ci(x, rng)

        row.update({
            "mean": mean,
            "std": std,
            "sem": std / np.sqrt(n) if n > 1 else np.nan,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "min": x.min(),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": x.max(),
            "iqr": q3 - q1,
            "range": x.max() - x.min(),
            "cv": std / abs(mean) if n > 1 and mean != 0 else np.nan,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def pooled_metrics(df):
    """
    Pooled (micro-averaged) measures: confusion counts are summed over
    all patients first, so large tumors weigh more than small ones.
    """
    needed = {
        "true_positive_voxels", "true_negative_voxels",
        "false_positive_voxels", "false_negative_voxels",
    }
    if not needed <= set(df.columns):
        return pd.DataFrame(columns=["metric", "value"])

    tp = df["true_positive_voxels"].sum()
    tn = df["true_negative_voxels"].sum()
    fp = df["false_positive_voxels"].sum()
    fn = df["false_negative_voxels"].sum()

    def ratio(a, b):
        return a / b if b else np.nan

    rows = [
        ("pooled_true_positive_voxels", tp),
        ("pooled_true_negative_voxels", tn),
        ("pooled_false_positive_voxels", fp),
        ("pooled_false_negative_voxels", fn),
        ("pooled_dice", ratio(2 * tp, 2 * tp + fp + fn)),
        ("pooled_iou_jaccard", ratio(tp, tp + fp + fn)),
        ("pooled_precision", ratio(tp, tp + fp)),
        ("pooled_sensitivity_recall", ratio(tp, tp + fn)),
        ("pooled_specificity", ratio(tn, tn + fp)),
        ("pooled_false_positive_rate", ratio(fp, fp + tn)),
        ("pooled_false_negative_rate", ratio(fn, fn + tp)),
        ("pooled_voxel_accuracy", ratio(tp + tn, tp + tn + fp + fn)),
    ]

    if {"predicted_volume_mm3", "ground_truth_volume_mm3"} <= set(df.columns):
        pred = df["predicted_volume_mm3"].sum()
        gt = df["ground_truth_volume_mm3"].sum()
        rows += [
            ("total_predicted_volume_mm3", pred),
            ("total_ground_truth_volume_mm3", gt),
            ("pooled_signed_relative_volume_error", ratio(pred - gt, gt)),
        ]

    return pd.DataFrame(rows, columns=["metric", "value"])


def pass_rates(df):
    rows = []
    n_dice = int(df["dice"].notna().sum())

    for t in DICE_THRESHOLDS:
        count = int((df["dice"] >= t).sum())
        rows.append({
            "criterion": f"Dice >= {t:.2f}",
            "patients_passing": count,
            "patients_evaluated": n_dice,
            "pass_rate_percent": 100 * count / n_dice if n_dice else np.nan,
        })

    if "hd95_mm" in df.columns:
        n_hd = int(df["hd95_mm"].notna().sum())
        for t in HD95_THRESHOLDS_MM:
            count = int((df["hd95_mm"] <= t).sum())
            rows.append({
                "criterion": f"HD95 <= {t} mm",
                "patients_passing": count,
                "patients_evaluated": n_hd,
                "pass_rate_percent": 100 * count / n_hd if n_hd else np.nan,
            })

    return pd.DataFrame(rows)


def find_outliers(df):
    rows = []
    for metric in OUTLIER_METRICS:
        if metric not in df.columns:
            continue
        x = df[metric].dropna()
        if len(x) < 4:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        for idx in x.index:
            value = df.loc[idx, metric]
            if value < low or value > high:
                rows.append({
                    "metric": metric,
                    "patient_id": df.loc[idx, "patient_id"],
                    "value": value,
                    "lower_fence": low,
                    "upper_fence": high,
                    "direction": "low" if value < low else "high",
                })
    return pd.DataFrame(
        rows,
        columns=["metric", "patient_id", "value",
                 "lower_fence", "upper_fence", "direction"],
    )


def correlation_table(df):
    columns = [c for c in CORRELATION_METRICS if c in df.columns]
    if len(columns) < 2:
        return pd.DataFrame()
    return df[columns].rank().corr()


def failure_mode_table(df):
    counts = df["failure_mode"].value_counts()
    total = counts.sum()
    out = counts.rename_axis("failure_mode").reset_index(name="patients")
    out["percent"] = 100 * out["patients"] / total
    out["patient_numbers"] = out["failure_mode"].apply(
        lambda m: ", ".join(df.loc[df["failure_mode"] == m, "patient_number"])
    )
    return out


def quality_tier_table(df):
    order = ["Excellent", "Good", "Moderate", "Poor", "No data"]
    counts = df["quality_tier"].value_counts()
    rows = []
    for tier in order:
        n = int(counts.get(tier, 0))
        rows.append({
            "quality_tier": tier,
            "patients": n,
            "percent": 100 * n / len(df),
            "patient_numbers": ", ".join(
                df.loc[df["quality_tier"] == tier, "patient_number"]
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# GRAPHS
# ============================================================

def save_figure(fig, folder, name):
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name), dpi=150)
    plt.close(fig)


def make_graphs(df, corr, folder):
    os.makedirs(folder, exist_ok=True)
    labels = df["patient_number"].tolist()
    x = np.arange(len(df))
    width = max(8, 0.45 * len(df) + 3)

    # 1. Dice & IoU per patient
    if {"dice", "iou_jaccard"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(x - 0.2, df["dice"], 0.4, label="Dice")
        ax.bar(x + 0.2, df["iou_jaccard"], 0.4, label="IoU")
        ax.axhline(df["dice"].mean(), color="k", ls="--", lw=1,
                   label=f"Mean Dice = {df['dice'].mean():.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Patient")
        ax.set_ylabel("Score")
        ax.set_title("Dice and IoU per patient")
        ax.legend()
        save_figure(fig, folder, "01_dice_iou_per_patient.png")

    # 2. Box plot of overlap metrics
    cols = [c for c in
            ["dice", "iou_jaccard", "precision", "sensitivity_recall"]
            if c in df.columns]
    if cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([df[c].dropna() for c in cols])
        ax.set_xticklabels(cols)
        ax.set_ylim(0, 1.05)
        ax.set_title("Distribution of overlap metrics")
        save_figure(fig, folder, "02_boxplot_overlap_metrics.png")

    # 3. Dice histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["dice"].dropna(), bins=np.linspace(0, 1, 21), edgecolor="k")
    ax.axvline(df["dice"].mean(), color="r", ls="--",
               label=f"Mean = {df['dice'].mean():.3f}")
    ax.axvline(df["dice"].median(), color="g", ls=":",
               label=f"Median = {df['dice'].median():.3f}")
    ax.set_xlabel("Dice")
    ax.set_ylabel("Number of patients")
    ax.set_title("Dice distribution")
    ax.legend()
    save_figure(fig, folder, "03_dice_histogram.png")

    # 4. Precision vs recall
    if {"precision", "sensitivity_recall"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(df["sensitivity_recall"], df["precision"])
        for lab, xv, yv in zip(labels, df["sensitivity_recall"],
                               df["precision"]):
            ax.annotate(lab, (xv, yv), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs recall\n"
                     "(below line = over-segmentation, above = under-segmentation)")
        save_figure(fig, folder, "04_precision_vs_recall.png")

    # 5. Predicted vs ground-truth volume
    if {"predicted_volume_mm3", "ground_truth_volume_mm3"} <= set(df.columns):
        gt = df["ground_truth_volume_mm3"] / 1000
        pr = df["predicted_volume_mm3"] / 1000
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(gt, pr)
        for lab, xv, yv in zip(labels, gt, pr):
            ax.annotate(lab, (xv, yv), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        top = float(np.nanmax([gt.max(), pr.max()])) * 1.05
        ax.plot([0, top], [0, top], "k--", lw=1, label="Perfect agreement")
        ax.set_xlabel("Ground-truth volume (cm$^3$)")
        ax.set_ylabel("Predicted volume (cm$^3$)")
        ax.set_title("Predicted vs ground-truth tumor volume")
        ax.legend()
        save_figure(fig, folder, "05_volume_predicted_vs_truth.png")

        # 6. Dice vs tumor size
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(gt, df["dice"])
        for lab, xv, yv in zip(labels, gt, df["dice"]):
            ax.annotate(lab, (xv, yv), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Ground-truth volume (cm$^3$)")
        ax.set_ylabel("Dice")
        ax.set_title("Dice vs tumor size")
        save_figure(fig, folder, "06_dice_vs_tumor_size.png")

    # 7. Surface distances
    if {"hd95_mm", "assd_mm"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(x - 0.2, df["hd95_mm"], 0.4, label="HD95 (mm)")
        ax.bar(x + 0.2, df["assd_mm"], 0.4, label="ASSD (mm)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlabel("Patient")
        ax.set_ylabel("Distance (mm)")
        ax.set_title("Boundary distance per patient")
        ax.legend()
        save_figure(fig, folder, "07_surface_distances.png")

    # 8. Signed relative volume error
    if "signed_relative_volume_error" in df.columns:
        err = 100 * df["signed_relative_volume_error"]
        colors = ["tab:red" if v > 0 else "tab:blue" for v in err.fillna(0)]
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(x, err, color=colors)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlabel("Patient")
        ax.set_ylabel("Signed volume error (%)")
        ax.set_title("Volume bias per patient "
                     "(red = over-segmented, blue = under-segmented)")
        save_figure(fig, folder, "08_signed_volume_error.png")

    # 9. Correlation heat map
    if corr is not None and not corr.empty:
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.index)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                value = corr.values[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center",
                            va="center", fontsize=7)
        fig.colorbar(image, ax=ax)
        ax.set_title("Spearman correlation between metrics")
        save_figure(fig, folder, "09_correlation_heatmap.png")


# ============================================================
# CONCLUSION
# ============================================================

def build_conclusion(df, summary, pooled, rates, outliers, corr,
                     start, end, missing, inf_counts):
    S = summary.set_index("metric")

    def stat(metric, column):
        if metric in S.index and column in S.columns:
            return S.loc[metric, column]
        return np.nan

    def mean_std(metric, digits=3):
        return f"{fmt(stat(metric, 'mean'), digits)} +/- {fmt(stat(metric, 'std'), digits)}"

    n = len(df)
    lines = []
    add = lines.append

    add("=" * 75)
    add(f" OVERALL PIPELINE CONCLUSION   ({make_patient_id(start)} -> {make_patient_id(end)})")
    add("=" * 75)
    add("")
    add(f"Patients analysed : {n} of {end - start + 1} in the requested range")
    if missing:
        add(f"Patients missing  : {len(missing)} "
            f"({', '.join(m[0][-PATIENT_NUMBER_WIDTH:] for m in missing)})")
    add(f"Region evaluated  : {', '.join(sorted(df['region'].dropna().unique())) if 'region' in df.columns else 'n/a'}")
    add("")

    # ---- headline -------------------------------------------
    mean_dice = stat("dice", "mean")
    add("-" * 75)
    add(" 1. HEADLINE RESULTS")
    add("-" * 75)
    add(f"Dice                : {mean_std('dice')}   "
        f"(median {fmt(stat('dice', 'median'), 3)}, "
        f"range {fmt(stat('dice', 'min'), 3)} - {fmt(stat('dice', 'max'), 3)})")
    add(f"  95% CI of mean    : {fmt(stat('dice', 'ci95_low'), 3)} - {fmt(stat('dice', 'ci95_high'), 3)}")
    add(f"IoU (Jaccard)       : {mean_std('iou_jaccard')}")
    add(f"Precision           : {mean_std('precision')}")
    add(f"Recall (sensitivity): {mean_std('sensitivity_recall')}")
    add(f"HD95 (mm)           : {mean_std('hd95_mm', 2)}   (median {fmt(stat('hd95_mm', 'median'), 2)})")
    add(f"ASSD (mm)           : {mean_std('assd_mm', 2)}   (median {fmt(stat('assd_mm', 'median'), 2)})")
    add(f"Volume similarity   : {mean_std('volume_similarity')}")

    pooled_dice = pooled.loc[pooled["metric"] == "pooled_dice", "value"]
    if len(pooled_dice):
        add(f"Pooled Dice         : {fmt(pooled_dice.iloc[0], 3)}  "
            "(all voxels combined; large tumors weigh more)")
    add("")
    add("Note: specificity and voxel accuracy are always very high "
        "(background dominates the volume),")
    add("so they say little about quality. Dice, precision/recall and "
        "HD95 are the meaningful numbers.")
    add("")

    # ---- verdict --------------------------------------------
    add("-" * 75)
    add(" 2. VERDICT")
    add("-" * 75)
    if pd.isna(mean_dice):
        verdict = "No Dice values available."
    elif mean_dice >= DICE_EXCELLENT:
        verdict = "STRONG - overlap is excellent on average."
    elif mean_dice >= DICE_GOOD:
        verdict = ("ACCEPTABLE - good overlap on average, which is respectable "
                   "for a classical (non-ML) pipeline, with room to improve.")
    elif mean_dice >= DICE_MODERATE:
        verdict = ("MODERATE - the pipeline finds the tumor but the extent is "
                   "often wrong; clear improvements are needed.")
    else:
        verdict = "WEAK - overlap with the ground truth is poor on average."
    add(verdict)
    add("(For context: classical methods on BraTS Whole Tumor usually fall "
        "between about 0.5 and 0.8 Dice;")
    add(" deep-learning models reach about 0.9.)")
    add("")

    # ---- distribution ---------------------------------------
    add("-" * 75)
    add(" 3. QUALITY DISTRIBUTION")
    add("-" * 75)
    for _, row in quality_tier_table(df).iterrows():
        if row["patients"]:
            add(f"{row['quality_tier']:<10}: {row['patients']:>3} patients "
                f"({row['percent']:.0f}%)   [{row['patient_numbers']}]")
    add("")
    for _, row in rates.iterrows():
        add(f"{row['criterion']:<14}: {int(row['patients_passing'])}/"
            f"{int(row['patients_evaluated'])} "
            f"({row['pass_rate_percent']:.0f}%)")
    add("")

    # ---- error behaviour ------------------------------------
    add("-" * 75)
    add(" 4. ERROR BEHAVIOUR")
    add("-" * 75)
    mean_p = stat("precision", "mean")
    mean_r = stat("sensitivity_recall", "mean")
    add(f"Mean precision {fmt(mean_p, 3)} vs mean recall {fmt(mean_r, 3)}.")

    tendency = None
    if not pd.isna(mean_p) and not pd.isna(mean_r):
        if mean_r - mean_p > 0.05:
            tendency = "over"
            add("-> The pipeline tends to OVER-segment: it captures the tumor "
                "but adds non-tumor voxels.")
        elif mean_p - mean_r > 0.05:
            tendency = "under"
            add("-> The pipeline tends to UNDER-segment: what it marks is "
                "mostly tumor, but it misses part of the tumor (often edema).")
        else:
            tendency = "balanced"
            add("-> Precision and recall are balanced; errors are not "
                "systematically in one direction.")

    if "signed_relative_volume_error" in df.columns:
        err = df["signed_relative_volume_error"].dropna()
        over = int((err > 0).sum())
        under = int((err < 0).sum())
        add(f"Volume bias: mean signed error {100 * err.mean():+.1f}% "
            f"(median {100 * err.median():+.1f}%); "
            f"{over} patients over-estimated, {under} under-estimated.")
    add("")

    add("Failure modes (patients below 'excellent' overlap):")
    for _, row in failure_mode_table(df).iterrows():
        add(f"  {row['failure_mode']:<44}: {row['patients']:>3} "
            f"[{row['patient_numbers']}]")
    add("")

    # ---- consistency & size effect --------------------------
    add("-" * 75)
    add(" 5. CONSISTENCY AND TUMOR-SIZE EFFECT")
    add("-" * 75)
    std_dice = stat("dice", "std")
    if not pd.isna(std_dice):
        if std_dice > 0.15:
            add(f"Dice std = {std_dice:.3f}: results vary a lot between patients, "
                "so one fixed parameter set does not fit all scans.")
        else:
            add(f"Dice std = {std_dice:.3f}: results are fairly consistent "
                "between patients.")

    if (not corr.empty and "dice" in corr.columns
            and "ground_truth_volume_mm3" in corr.columns):
        rho = corr.loc["dice", "ground_truth_volume_mm3"]
        if not pd.isna(rho):
            if rho > 0.4:
                add(f"Dice vs tumor size: Spearman rho = {rho:.2f} -> the pipeline "
                    "does WORSE on small tumors.")
            elif rho < -0.4:
                add(f"Dice vs tumor size: Spearman rho = {rho:.2f} -> the pipeline "
                    "does WORSE on large tumors.")
            else:
                add(f"Dice vs tumor size: Spearman rho = {rho:.2f} -> no strong "
                    "dependence on tumor size.")

    hd_median = stat("hd95_mm", "median")
    if not pd.isna(hd_median):
        far = int((df["hd95_mm"] > 20).sum())
        add(f"Median HD95 = {hd_median:.1f} mm; {far} patient(s) above 20 mm "
            "(large distant errors such as stray false-positive regions).")
    add("")

    # ---- best / worst ---------------------------------------
    add("-" * 75)
    add(" 6. BEST AND WORST CASES (by Dice)")
    add("-" * 75)
    ranked = df.sort_values("dice", ascending=False)
    k = min(3, len(ranked))
    add("Best : " + ", ".join(
        f"{r.patient_number} ({r.dice:.3f})" for r in ranked.head(k).itertuples()))
    add("Worst: " + ", ".join(
        f"{r.patient_number} ({r.dice:.3f}, {r.failure_mode})"
        for r in ranked.tail(k).iloc[::-1].itertuples()))
    if not outliers.empty:
        add("")
        add("Statistical outliers (1.5 x IQR rule): " + "; ".join(
            f"{p}: {', '.join(g['metric'] + ' ' + g['direction'] for _, g in grp.iterrows())}"
            for p, grp in outliers.groupby("patient_id")))
    add("")

    # ---- recommendations ------------------------------------
    add("-" * 75)
    add(" 7. SUGGESTED NEXT STEPS")
    add("-" * 75)
    tips = []
    if tendency == "under":
        tips.append("Recall is the weak point: lower the threshold or use "
                    "hysteresis thresholding / region growing, and rely on "
                    "FLAIR or T2 (not T1ce) for the Whole Tumor so edema "
                    "is included.")
    if tendency == "over":
        tips.append("Precision is the weak point: add or tighten skull "
                    "stripping / brain masking and keep only the connected "
                    "components that touch the tumor seed.")
    if not pd.isna(std_dice) and std_dice > 0.15:
        tips.append("Normalize intensities per patient (z-score or percentile "
                    "inside a brain mask) so a single threshold behaves the "
                    "same on every scan.")
    if not pd.isna(hd_median) and (df["hd95_mm"] > 20).any():
        tips.append("Large HD95 values point to distant false-positive blobs; "
                    "connected-component filtering and morphological opening "
                    "should remove them.")
    tips.append("Re-run this script after each pipeline change (bump "
                "PIPELINE_VERSION) and compare the mean Dice, HD95 and "
                "failure-mode counts.")
    for i, tip in enumerate(tips, start=1):
        add(f"{i}. {tip}")
    add("")

    # ---- data notes -----------------------------------------
    if inf_counts:
        add("-" * 75)
        add(" DATA NOTES")
        add("-" * 75)
        for metric, count in inf_counts.items():
            add(f"{metric}: {count} infinite value(s) treated as missing.")
        add("")

    add("=" * 75)
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    start, end = get_range()

    range_name = f"{start:0{PATIENT_NUMBER_WIDTH}d}-{end:0{PATIENT_NUMBER_WIDTH}d}"
    out_dir = os.path.join(STATS_ROOT, range_name)
    graph_dir = os.path.join(out_dir, "Graphs")
    os.makedirs(graph_dir, exist_ok=True)

    print()
    print(f"Reading segmentation-evaluation tables from: {TABLES_ROOT}")

    records, missing, notes = load_range(start, end)

    for patient_id, note in missing:
        print(f"[SKIPPED] {patient_id}: {note}")
    for patient_id, note in notes:
        print(f"[NOTE]    {patient_id}: {note}")

    if not records:
        print()
        print("No segmentation-evaluation CSVs found in this range. Nothing to do.")
        return

    df, inf_counts = build_patient_table(records)
    metrics = numeric_metric_columns(df)

    summary = summarize_metrics(df, metrics)
    pooled = pooled_metrics(df)
    rates = pass_rates(df)
    outliers = find_outliers(df)
    corr = correlation_table(df)
    failures = failure_mode_table(df)
    tiers = quality_tier_table(df)

    # ---- per-patient tables ---------------------------------
    front = ["patient_id", "patient_number", "run_id", "region",
             "quality_tier", "failure_mode"]
    front = [c for c in front if c in df.columns]
    table = df[front + metrics + ["source_file"]]

    table.to_csv(os.path.join(out_dir, "per_patient_metrics.csv"), index=False)

    ranking = table.sort_values("dice", ascending=False).copy()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking[["rank", "patient_id", "quality_tier", "failure_mode", "dice",
             "iou_jaccard", "precision", "sensitivity_recall"]
            + [c for c in ["hd95_mm", "assd_mm", "signed_relative_volume_error"]
               if c in ranking.columns]
            ].to_csv(os.path.join(out_dir, "patient_ranking.csv"), index=False)

    # ---- statistics tables ----------------------------------
    summary.to_csv(os.path.join(out_dir, "summary_statistics.csv"), index=False)
    pooled.to_csv(os.path.join(out_dir, "pooled_metrics.csv"), index=False)
    rates.to_csv(os.path.join(out_dir, "threshold_pass_rates.csv"), index=False)
    outliers.to_csv(os.path.join(out_dir, "outliers.csv"), index=False)
    failures.to_csv(os.path.join(out_dir, "failure_modes.csv"), index=False)
    tiers.to_csv(os.path.join(out_dir, "quality_tiers.csv"), index=False)
    if not corr.empty:
        corr.to_csv(os.path.join(out_dir, "correlation_matrix_spearman.csv"))

    # ---- graphs ---------------------------------------------
    make_graphs(df, corr, graph_dir)

    # ---- conclusion -----------------------------------------
    conclusion = build_conclusion(
        df, summary, pooled, rates, outliers, corr,
        start, end, missing, inf_counts,
    )

    with open(os.path.join(out_dir, "pipeline_conclusion.txt"),
              "w", encoding="utf-8") as handle:
        handle.write(conclusion + "\n")

    print()
    print(conclusion)
    print()
    print(f"All range statistics saved to: {out_dir}")


if __name__ == "__main__":
    main()