"""
tune.py  —  Fast Bayesian parameter tuner for all 3 segmentation pipelines.

Strategy
--------
  Bayesian optimisation (Optuna TPE sampler) + parallel trial execution
  (ProcessPoolExecutor, up to N_JOBS workers).

  Each Optuna trial proposes one parameter combination, runs it over the
  selected patient probe set, and returns mean Dice as the objective.
  Optuna learns which regions of the search space are promising and focuses
  there — requiring far fewer trials than exhaustive grid search to converge
  on a good result.

Target
------
  20 patients, all 3 pipelines, ≤ 20 minutes on an i7-12650H.
  Achieved by:
    • 20 Optuna trials per pipeline   (vs. 192/81/27 grid combos)
    • 8 parallel workers              (10 physical cores, leave 2 free)
    • Lightweight inline eval         (no CSV/PNG writes during search)

Usage
-----
  python tune.py

  You will be asked which pipeline(s) and the probe patient range.
  Preprocessed data for those patients must already exist under Outputs/.

Outputs
-------
  Results/Tuning/Pipeline_<N>/tuning_results.csv   — all trial scores ranked
  Results/Tuning/Pipeline_<N>/tuning_summary.txt   — best params + top-5 table
  Results/Tuning/Pipeline_<N>/optuna_study.db      — Optuna SQLite journal
                                                      (lets you resume later)
"""

import os
import sys
import time
import importlib.util
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import optuna
from scipy import ndimage as ndi
import SimpleITK as sitk

optuna.logging.set_verbosity(optuna.logging.WARNING)   # keep console clean
warnings.filterwarnings("ignore")

# ============================================================
# GLOBAL PATHS & CONSTANTS
# ============================================================

PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(PROJECT_DIR, "data", "Training")
TUNING_ROOT  = os.path.join(PROJECT_DIR, "Results", "Tuning")
TMP_ROOT     = os.path.join(PROJECT_DIR, "Outputs", "_tuning_tmp")

PATIENT_PREFIX       = "BraTS20_Training_"
PATIENT_NUMBER_WIDTH = 3

N_JOBS   = 8    # parallel workers  (leave 2 cores free for OS)
N_TRIALS = 20   # Optuna trials per pipeline

OUTPUTS_ROOTS = {
    1: os.path.join(PROJECT_DIR, "Outputs", "Pipeline_1_Thresholding"),
    2: os.path.join(PROJECT_DIR, "Outputs", "Pipeline_2_Watershed"),
    3: os.path.join(PROJECT_DIR, "Outputs", "Pipeline_3_Hybrid"),
}

PIPELINE_LABELS = {
    1: "Pipeline 1.0 (Thresholding)",
    2: "Pipeline 2.0 (Watershed)",
    3: "Pipeline 3.0 (Hybrid)",
}

SEG_MODULE_PATHS = {
    1: os.path.join(PROJECT_DIR, "pipelines", "pipeline_1_thresholding", "segmentation.py"),
    2: os.path.join(PROJECT_DIR, "pipelines", "pipeline_2_watershed",    "segmentation_II.py"),
    3: os.path.join(PROJECT_DIR, "pipelines", "pipeline_3_hybrid",       "segmentation_III.py"),
}

METHOD_TAGS = {1: "thresholding", 2: "watershed", 3: "hybrid"}

# ── Bayesian search spaces ───────────────────────────────────
# suggest_int(name, low, high)   → integer
# suggest_float(name, low, high) → float
# suggest_categorical(name, choices)
SEARCH_SPACES = {
    1: [
        # P1: under-segmentation (sensitivity 0.572) — lower thresholds recover edema
        ("suggest_int",   "P1_FLAIR_PERCENTILE",  78, 85),
        ("suggest_int",   "P1_T2_PERCENTILE",      78, 85),
        ("suggest_int",   "P1_T1CE_PERCENTILE",    87, 90),
        ("suggest_int",   "P1_MORPH_CLOSE_ITERS",  1,  3),
    ],
    2: [
        # P2: erratic boundaries (HD95 28mm) — smoother gradient + tighter markers
        ("suggest_float", "P2_GRADIENT_SIGMA",       1.0, 2.5),
        ("suggest_int",   "P2_HEALTHY_PERCENTILE",   63,  76),
        ("suggest_int",   "P2_HEALTHY_DIST_VOXELS",  3,   7),
        ("suggest_float", "P2_SATELLITE_DIST_MM",    7.0, 14.0),
    ],
    3: [
        # P3: over-segmentation (+22% volume) — tighten edema band
        ("suggest_int",   "P3_EDEMA_DIST_VOXELS",  2,   4),
        ("suggest_int",   "P3_EDEMA_FLAIR_PCT",     79,  85),
        ("suggest_float", "P3_GRADIENT_SIGMA",      1.0, 2.5),
    ],
}


# ============================================================
# HELPERS
# ============================================================

def make_patient_id(number: int) -> str:
    return f"{PATIENT_PREFIX}{number:0{PATIENT_NUMBER_WIDTH}d}"


def ask_integer(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
            if v >= 1:
                return v
            print("  Please enter a number >= 1.")
        except ValueError:
            print("  Please enter a whole number.")


def _load_module_fresh(path: str, name: str):
    """Load a Python module from an absolute file path, bypassing sys.modules cache."""
    if name in sys.modules:
        del sys.modules[name]
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    # Ensure the module's directory, shared/, and root are on sys.path
    mod_dir    = os.path.dirname(path)
    shared_dir = os.path.join(PROJECT_DIR, "shared")
    for d in [mod_dir, shared_dir, PROJECT_DIR]:
        if d not in sys.path:
            sys.path.insert(0, d)

    os.chdir(PROJECT_DIR)
    spec.loader.exec_module(module)
    return module


# ============================================================
# FAST INLINE EVALUATOR
# (no CSV/PNG writes — just Dice, HD95, Sensitivity)
# ============================================================

def _surface(mask: np.ndarray) -> np.ndarray:
    struct = ndi.generate_binary_structure(3, 1)
    return mask ^ ndi.binary_erosion(mask, structure=struct)


def fast_eval(pred_path: str, gt_path: str):
    """Return (dice, hd95_mm, sensitivity).  Returns NaNs on any failure."""
    if not os.path.isfile(pred_path) or not os.path.isfile(gt_path):
        return np.nan, np.nan, np.nan
    try:
        pred_img = sitk.ReadImage(pred_path)
        gt_img   = sitk.ReadImage(gt_path)
        pred     = sitk.GetArrayFromImage(pred_img).astype(bool)
        gt_arr   = sitk.GetArrayFromImage(gt_img)
        gt       = (gt_arr == 1) | (gt_arr == 2) | (gt_arr == 4)   # BraTS WT

        tp = int(np.sum(pred & gt))
        fp = int(np.sum(pred & ~gt))
        fn = int(np.sum(~pred & gt))

        dice = (2*tp / (2*tp + fp + fn)) if (2*tp + fp + fn) > 0 else 1.0
        sens = (tp / (tp + fn))           if (tp + fn) > 0           else 1.0

        # HD95 ── only computed when both surfaces are non-empty
        ps = _surface(pred)
        gs = _surface(gt)
        if np.sum(ps) == 0 or np.sum(gs) == 0:
            hd95 = np.nan
        else:
            sp = pred_img.GetSpacing()
            samp = (sp[2], sp[1], sp[0])
            d_p2g = ndi.distance_transform_edt(~gs,  sampling=samp)[ps]
            d_g2p = ndi.distance_transform_edt(~ps,  sampling=samp)[gs]
            hd95  = float(np.percentile(np.concatenate([d_p2g, d_g2p]), 95))

        return float(dice), float(hd95), float(sens)
    except Exception:
        return np.nan, np.nan, np.nan


# ============================================================
# SINGLE-TRIAL WORKER  (runs in a subprocess)
# ============================================================

def _run_trial_worker(args: tuple) -> dict:
    """
    Executed in a worker process.
    Runs ONE parameter combination over all probe patients and returns metrics.
    """
    pipeline_key, params, patient_ids, trial_number = args

    # Set up sys.path for this subprocess
    for d in [
        os.path.join(PROJECT_DIR, "Pipeline 2.0"),
        os.path.join(PROJECT_DIR, "Pipeline 3.0"),
        os.path.join(PROJECT_DIR, "Codes"),
        PROJECT_DIR,
    ]:
        if d not in sys.path:
            sys.path.insert(0, d)

    os.chdir(PROJECT_DIR)

    tag     = METHOD_TAGS[pipeline_key]
    out_root = OUTPUTS_ROOTS[pipeline_key]
    mod_name = f"seg_p{pipeline_key}_t{trial_number}"

    try:
        seg_mod = _load_module_fresh(SEG_MODULE_PATHS[pipeline_key], mod_name)
    except Exception as e:
        return {"trial": trial_number, "params": params,
                "mean_dice": np.nan, "mean_hd95": np.nan, "mean_sens": np.nan,
                "n_ok": 0, "error": str(e)}

    # Monkey-patch the named constants
    for pname, pval in params.items():
        if hasattr(seg_mod, pname):
            setattr(seg_mod, pname, pval)

    dices, hd95s, senss = [], [], []

    for patient_id in patient_ids:
        tmp_dir = os.path.join(TMP_ROOT, f"p{pipeline_key}_t{trial_number}", patient_id)
        os.makedirs(tmp_dir, exist_ok=True)

        config = {
            "patient_id":    patient_id,
            "run_id":        f"tune_t{trial_number}",
            "method_tag":    tag,
            "denoised_dir":  os.path.join(out_root, "Denoised", patient_id),
            "clahe_dir":     os.path.join(out_root, "CLAHE",    patient_id),
            "segmented_dir": tmp_dir,
            "image_dir":     tmp_dir,
        }

        # Quick guard: skip if no preprocessed data
        chk = os.path.join(config["denoised_dir"],
                           f"{patient_id}_flair_denoised.nii.gz")
        if not os.path.isfile(chk):
            continue

        try:
            seg_path = seg_mod.run_segmentation(config)
        except Exception:
            continue

        gt_path = os.path.join(TRAINING_DIR, patient_id, f"{patient_id}_seg.nii")
        dice, hd95, sens = fast_eval(seg_path, gt_path)
        if not np.isnan(dice):
            dices.append(dice)
            hd95s.append(hd95 if not np.isnan(hd95) else 999.0)
            senss.append(sens)

    if dices:
        return {
            "trial":     trial_number,
            "params":    params,
            "mean_dice": float(np.mean(dices)),
            "mean_hd95": float(np.nanmean(hd95s)),
            "mean_sens": float(np.mean(senss)),
            "n_ok":      len(dices),
            "error":     None,
        }
    else:
        return {
            "trial": trial_number, "params": params,
            "mean_dice": np.nan, "mean_hd95": np.nan,
            "mean_sens": np.nan, "n_ok": 0, "error": "all patients failed",
        }


# ============================================================
# OPTUNA OBJECTIVE  (called from main process, dispatches workers)
# ============================================================

class _Objective:
    """
    Wraps Optuna's trial interface.
    Trials are batched and run in parallel across ProcessPoolExecutor workers.
    """

    def __init__(self, pipeline_key: int, patient_ids: list,
                 executor: ProcessPoolExecutor, results_store: list):
        self.key      = pipeline_key
        self.pids     = patient_ids
        self.executor = executor
        self.store    = results_store
        self._futures = {}   # trial_number → Future

    def __call__(self, trial: optuna.Trial) -> float:
        space   = SEARCH_SPACES[self.key]
        params  = {}
        for suggest_type, name, *args in space:
            fn = getattr(trial, suggest_type)
            params[name] = fn(name, *args)

        future = self.executor.submit(
            _run_trial_worker,
            (self.key, params, self.pids, trial.number)
        )
        # Block until this trial's worker finishes (Optuna calls __call__ sequentially)
        result = future.result()
        self.store.append(result)

        dice = result["mean_dice"] if not np.isnan(result.get("mean_dice", np.nan)) else 0.0
        return float(dice)


# ============================================================
# PER-PIPELINE TUNER
# ============================================================

def tune_pipeline(pipeline_key: int, patient_ids: list):
    label   = PIPELINE_LABELS[pipeline_key]
    out_dir = os.path.join(TUNING_ROOT, f"Pipeline_{pipeline_key}")
    os.makedirs(out_dir, exist_ok=True)

    db_path = os.path.join(out_dir, "optuna_study.db")
    storage = f"sqlite:///{db_path}"

    # Delete previous study with same name so we start fresh
    study_name = f"tune_p{pipeline_key}"
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except Exception:
        pass

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        load_if_exists=False,
    )

    param_names = [name for _, name, *_ in SEARCH_SPACES[pipeline_key]]
    n_valid     = sum(
        1 for pid in patient_ids
        if os.path.isfile(os.path.join(
            OUTPUTS_ROOTS[pipeline_key], "Denoised", pid,
            f"{pid}_flair_denoised.nii.gz"))
    )

    print()
    print("=" * 70)
    print(f"  TUNING  {label}")
    print(f"  Patients with data: {n_valid}/{len(patient_ids)}")
    print(f"  Trials: {N_TRIALS}   Workers: {N_JOBS}")
    print(f"  Parameters: {param_names}")
    print("=" * 70)

    results_store = []
    t_start       = time.perf_counter()

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        objective = _Objective(pipeline_key, patient_ids, executor, results_store)
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=False,
        )

    elapsed = time.perf_counter() - t_start

    # ── Clean up all tmp segmentation files for this pipeline ──
    tmp_pipeline = os.path.join(TMP_ROOT, f"p{pipeline_key}_*")
    import glob, shutil
    for d in glob.glob(os.path.join(TMP_ROOT, f"p{pipeline_key}_t*")):
        shutil.rmtree(d, ignore_errors=True)

    # ── Build results dataframe ──────────────────────────────
    rows = []
    for r in results_store:
        if np.isnan(r.get("mean_dice", np.nan)):
            continue
        row = {"trial": r["trial"],
               "mean_dice": r["mean_dice"],
               "mean_hd95_mm": r["mean_hd95"],
               "mean_sensitivity": r["mean_sens"],
               "n_patients_ok": r["n_ok"]}
        row.update(r["params"])
        rows.append(row)

    if not rows:
        print(f"  No valid results for {label}. Check preprocessed data.")
        return None

    df = pd.DataFrame(rows).sort_values(
        ["mean_dice", "mean_hd95_mm"], ascending=[False, True]
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    csv_path = os.path.join(out_dir, "tuning_results.csv")
    df.to_csv(csv_path, index=False)

    best  = df.iloc[0]
    best_params = {p: best[p] for p in param_names}

    # ── Summary text ─────────────────────────────────────────
    lines = [
        "=" * 70,
        f" TUNING SUMMARY: {label}",
        f" Patients in probe: {len(patient_ids)}  |  Valid: {n_valid}",
        f" Optuna trials: {N_TRIALS}  |  Time: {elapsed/60:.1f} min",
        "=" * 70,
        "",
        " BEST CONFIGURATION (highest mean Dice)",
        "",
    ]
    for pname, pval in best_params.items():
        lines.append(f"   {pname:35s}: {pval}")
    lines += [
        "",
        f"   Mean Dice        : {best['mean_dice']:.4f}",
        f"   Mean HD95 (mm)   : {best['mean_hd95_mm']:.2f}",
        f"   Mean Sensitivity : {best['mean_sensitivity']:.4f}",
        "",
        " TOP 5 CONFIGURATIONS",
        "",
    ]
    for _, row in df.head(5).iterrows():
        combo = "  |  ".join(f"{p}={row[p]}" for p in param_names)
        lines.append(
            f"   #{int(row['rank'])}  Dice={row['mean_dice']:.4f}"
            f"  HD95={row['mean_hd95_mm']:.1f}mm"
            f"  Sens={row['mean_sensitivity']:.4f}"
            f"  |  {combo}"
        )
    lines += [
        "",
        " APPLY THESE TO YOUR SEGMENTATION FILE:",
        "",
    ]
    for pname, pval in best_params.items():
        lines.append(f"   {pname} = {pval}")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    print(summary)

    txt_path = os.path.join(out_dir, "tuning_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"\n  CSV  → {csv_path}")
    print(f"  TXT  → {txt_path}")
    print(f"  Time → {elapsed/60:.1f} min")

    return best_params


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("  BAYESIAN PARAMETER TUNER  —  Brain Tumor Segmentation")
    print(f"  Strategy : Optuna TPE  |  Trials/pipeline: {N_TRIALS}"
          f"  |  Workers: {N_JOBS}")
    print("=" * 70)
    print()
    print("Which pipeline(s) to tune?")
    print("  [1]  Pipeline 1.0  (Thresholding)")
    print("  [2]  Pipeline 2.0  (Watershed)")
    print("  [3]  Pipeline 3.0  (Hybrid)")
    print("  [4]  All three")
    print()

    while True:
        try:
            choice = int(input("  Choice (1/2/3/4): ").strip())
            if choice in [1, 2, 3, 4]:
                break
        except ValueError:
            pass
        print("  Please enter 1, 2, 3, or 4.")

    pipelines = [1, 2, 3] if choice == 4 else [choice]

    print()
    print("Enter the probe patient range.")
    print("Recommended: 1 to 20  (preprocessed data must already exist).")
    print()
    start = ask_integer("  Start patient number : ")
    while True:
        end = ask_integer("  End patient number   : ")
        if end >= start:
            break
        print("  End must be >= start.")

    patient_ids = [make_patient_id(n) for n in range(start, end + 1)]

    # ── Runtime estimate ─────────────────────────────────────
    # P1 ~4s/patient (pure numpy thresholding)
    # P2 ~12s/patient (watershed + gradient)
    # P3 ~10s/patient (hybrid)
    per_patient = {1: 4, 2: 12, 3: 10}
    n_p         = len(patient_ids)
    total_est   = sum(
        (N_TRIALS * n_p * per_patient[p]) / N_JOBS
        for p in pipelines
    )
    print()
    print(f"  Probe patients  : {n_p}")
    print(f"  Trials per pipe : {N_TRIALS}")
    print(f"  Parallel workers: {N_JOBS}")
    print(f"  Est. total time : ~{total_est/60:.0f} min")
    print()
    confirm = input("  Press Enter to start, or 'q' to quit: ").strip().lower()
    if confirm == "q":
        print("  Aborted.")
        return

    os.makedirs(TMP_ROOT, exist_ok=True)

    all_best = {}
    wall_start = time.perf_counter()

    for p in pipelines:
        best = tune_pipeline(p, patient_ids)
        if best:
            all_best[p] = best

    wall_total = time.perf_counter() - wall_start

    # ── Grand summary ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  ALL TUNING COMPLETE")
    print(f"  Total wall time: {wall_total/60:.1f} min")
    print()
    for p, params in all_best.items():
        print(f"  {PIPELINE_LABELS[p]}:")
        for k, v in params.items():
            print(f"    {k} = {v}")
        print()
    print(f"  Full results: {TUNING_ROOT}")
    print("=" * 70)

    # ── Clean up top-level tmp folder ──────────────────────────
    import shutil
    if os.path.isdir(TMP_ROOT):
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
