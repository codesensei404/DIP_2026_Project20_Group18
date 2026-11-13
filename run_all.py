"""
run_all.py  —  Unified launcher for all three segmentation pipelines.

Usage:
    python run_all.py

Asks for:
  1. Patient range  (e.g. 1 → 100)
  2. Which pipeline(s):
       [1] Pipeline 1.0  — Thresholding
       [2] Pipeline 2.0  — Watershed
       [3] Pipeline 3.0  — Hybrid

Each pipeline runs independently over the same patient range.
Outputs and Results are written to their respective subfolders under
Outputs/ and Results/ — nothing is shared or overwritten between pipelines.

Project layout
--------------
  pipelines/
    pipeline_1_thresholding/   main_p1.py + pipeline modules
    pipeline_2_watershed/      main_p2.py + pipeline modules
    pipeline_3_hybrid/         main_p3.py + pipeline modules
  shared/
    segmentation_evaluation.py
    range_statistics.py
  Outputs/
    Pipeline_1_Thresholding/   Denoised/ CLAHE/ Segmented/
    Pipeline_2_Watershed/      Denoised/ CLAHE/ Segmented/
    Pipeline_3_Hybrid/         Denoised/ CLAHE/ Segmented/
  Results/
    Pipeline_1_Thresholding/   Images/ Tables/ Graphs/ Range_Statistics/
    Pipeline_2_Watershed/      Images/ Tables/ Graphs/ Range_Statistics/
    Pipeline_3_Hybrid/         Images/ Tables/ Graphs/ Range_Statistics/
    Comparative_Analysis/
    Tuning/
"""

import os
import sys
import traceback
import importlib.util

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(PROJECT_DIR, "data", "Training")

PIPELINE_CONFIGS = {
    1: {
        "label":       "Pipeline 1.0  (Thresholding)",
        "main_file":   os.path.join(PROJECT_DIR, "pipelines", "pipeline_1_thresholding", "main_p1.py"),
        "module_name": "pipeline1_main",
    },
    2: {
        "label":       "Pipeline 2.0  (Watershed)",
        "main_file":   os.path.join(PROJECT_DIR, "pipelines", "pipeline_2_watershed",    "main_p2.py"),
        "module_name": "pipeline2_main",
    },
    3: {
        "label":       "Pipeline 3.0  (Hybrid)",
        "main_file":   os.path.join(PROJECT_DIR, "pipelines", "pipeline_3_hybrid",       "main_p3.py"),
        "module_name": "pipeline3_main",
    },
}

PATIENT_PREFIX       = "BraTS20_Training_"
PATIENT_NUMBER_WIDTH = 3

# ============================================================
# HELPERS
# ============================================================

def make_patient_id(number: int) -> str:
    return f"{PATIENT_PREFIX}{number:0{PATIENT_NUMBER_WIDTH}d}"


def ask_integer(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if value >= 1:
                return value
            print("  Please enter a number >= 1.")
        except ValueError:
            print("  Please enter a whole number.")


def ask_patient_range() -> tuple:
    print()
    print("Enter the patient range to process.")
    print(f"  Example: 1 to 100  →  {make_patient_id(1)} … {make_patient_id(100)}")
    print()
    start = ask_integer("  Start patient number : ")
    while True:
        end = ask_integer("  End patient number   : ")
        if end >= start:
            return start, end
        print("  End must be >= start.")


def ask_pipelines() -> list:
    print()
    print("Select which pipeline(s) to run.")
    print("  Enter numbers separated by spaces, or press Enter for all three.")
    print()
    for key, cfg in PIPELINE_CONFIGS.items():
        print(f"    [{key}]  {cfg['label']}")
    print()
    while True:
        raw = input("  Your choice (e.g.  1  or  1 3  or  1 2 3): ").strip()
        if raw == "":
            chosen = sorted(PIPELINE_CONFIGS.keys())
            print(f"  → Running all pipelines: {chosen}")
            return chosen
        try:
            chosen = sorted({int(p) for p in raw.split()})
        except ValueError:
            print("  Invalid input — enter numbers like: 1  or  2 3  or  1 2 3")
            continue
        invalid = [k for k in chosen if k not in PIPELINE_CONFIGS]
        if invalid:
            print(f"  Unknown pipeline(s): {invalid}. Valid choices are 1, 2, 3.")
            continue
        return chosen


def load_pipeline_module(pipeline_key: int):
    cfg         = PIPELINE_CONFIGS[pipeline_key]
    main_file   = cfg["main_file"]
    module_name = cfg["module_name"]

    if not os.path.isfile(main_file):
        raise FileNotFoundError(f"Pipeline main file not found:\n  {main_file}")

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec   = importlib.util.spec_from_file_location(module_name, main_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================
# PER-PIPELINE BATCH RUNNER
# ============================================================

def run_pipeline_for_range(pipeline_key: int, start: int, end: int, module):
    label   = PIPELINE_CONFIGS[pipeline_key]["label"]
    total   = end - start + 1
    completed, skipped, failed = [], [], []

    for position, number in enumerate(range(start, end + 1), start=1):
        patient_id = make_patient_id(number)
        data_dir   = os.path.join(TRAINING_DIR, patient_id)

        if not os.path.isdir(data_dir):
            print(f"\n  [SKIPPED]  {patient_id}  — data folder not found")
            skipped.append(patient_id)
            continue

        try:
            config = module.build_config(patient_id)
            print()
            print("-" * 75)
            print(f"  {label}  |  Patient {position}/{total}  |  {patient_id}")
            print("-" * 75)
            module.run_pipeline(config)
            completed.append(patient_id)
        except KeyboardInterrupt:
            print("\n  Interrupted by user. Stopping this pipeline.")
            break
        except Exception as error:
            print(f"\n  [FAILED]  {patient_id}: {type(error).__name__}: {error}")
            traceback.print_exc()
            failed.append(patient_id)

    return completed, skipped, failed


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 75)
    print("       BRAIN TUMOR SEGMENTATION  —  UNIFIED LAUNCHER")
    print("=" * 75)
    print()
    for key, cfg in PIPELINE_CONFIGS.items():
        print(f"    [{key}]  {cfg['label']}")

    start, end = ask_patient_range()
    pipelines  = ask_pipelines()

    print()
    print("=" * 75)
    print("  CONFIGURATION SUMMARY")
    print("=" * 75)
    print(f"  Patient range  :  {make_patient_id(start)}  →  {make_patient_id(end)}")
    print(f"  Total patients :  {end - start + 1}")
    print(f"  Pipelines      :  {[PIPELINE_CONFIGS[k]['label'] for k in pipelines]}")
    print()

    confirm = input("  Press Enter to start, or 'q' to quit: ").strip().lower()
    if confirm == "q":
        print("  Aborted.")
        return

    # Load all modules up front so import errors surface before any work starts
    modules = {}
    for key in pipelines:
        print(f"\n  Loading {PIPELINE_CONFIGS[key]['label']} …")
        try:
            modules[key] = load_pipeline_module(key)
            print("  OK")
        except Exception as error:
            print(f"  [ERROR] Could not load pipeline {key}: {error}")
            traceback.print_exc()
            return

    # Run
    overall = {}
    for key in pipelines:
        label = PIPELINE_CONFIGS[key]["label"]
        print()
        print("=" * 75)
        print(f"  STARTING  {label.upper()}")
        print(f"  Range: {make_patient_id(start)}  →  {make_patient_id(end)}")
        print("=" * 75)

        completed, skipped, failed = run_pipeline_for_range(key, start, end, modules[key])
        overall[key] = {"label": label, "completed": completed,
                        "skipped": skipped, "failed": failed}

        print()
        print("=" * 75)
        print(f"  {label.upper()}  —  DONE")
        print(f"  Completed: {len(completed)}  |  Skipped: {len(skipped)}  |  Failed: {len(failed)}")
        print("=" * 75)

    # Grand summary
    print()
    print("=" * 75)
    print("                    OVERALL BATCH SUMMARY")
    print("=" * 75)
    print(f"  Patient range  :  {make_patient_id(start)}  →  {make_patient_id(end)}")
    print()
    for key, info in overall.items():
        print(f"  {info['label']}")
        print(f"    Completed : {len(info['completed'])}")
        print(f"    Skipped   : {len(info['skipped'])}")
        print(f"    Failed    : {len(info['failed'])}")
        if info["skipped"]: print(f"    Skipped   : {', '.join(info['skipped'])}")
        if info["failed"]:  print(f"    Failed    : {', '.join(info['failed'])}")
        print()
    print("=" * 75)


if __name__ == "__main__":
    main()
