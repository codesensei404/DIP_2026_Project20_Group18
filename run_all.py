"""
run_all.py  —  Unified launcher for all three segmentation pipelines.

Usage:
    python run_all.py

The script will ask you:
  1. The patient range to process  (e.g. 1 → 20)
  2. Which pipeline(s) to run:
       [1] Pipeline 1.0  — Thresholding
       [2] Pipeline 2.0  — Watershed
       [3] Pipeline 3.0  — Hybrid (Threshold-Guided Watershed)
     You can pick one, two, or all three (e.g. "1", "1 3", "1 2 3").

Each selected pipeline processes the same patient range independently.
Results land in their respective Outputs/ and Results/ subfolders so
nothing is ever overwritten across pipelines.
"""

import os
import sys
import traceback
import importlib.util


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_CONFIGS = {
    1: {
        "label":       "Pipeline 1.0  (Thresholding)",
        "main_file":   os.path.join(PROJECT_DIR, "main.py"),
        "module_name": "pipeline1_main",
    },
    2: {
        "label":       "Pipeline 2.0  (Watershed)",
        "main_file":   os.path.join(PROJECT_DIR, "Pipeline 2.0", "main_II.py"),
        "module_name": "pipeline2_main",
    },
    3: {
        "label":       "Pipeline 3.0  (Hybrid)",
        "main_file":   os.path.join(PROJECT_DIR, "Pipeline 3.0", "main_III.py"),
        "module_name": "pipeline3_main",
    },
}

PATIENT_PREFIX       = "BraTS20_Training_"
PATIENT_NUMBER_WIDTH = 3
TRAINING_DIR         = os.path.join(PROJECT_DIR, "data", "Training")


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
            print("  Please enter a number of 1 or more.")
        except ValueError:
            print("  Please enter a whole number.")


def ask_patient_range() -> tuple:
    print()
    print("Enter the patient range to process.")
    print(f"  Example: 1 to 10  →  {make_patient_id(1)} … {make_patient_id(10)}")
    print()

    start = ask_integer("  Start patient number : ")

    while True:
        end = ask_integer("  End patient number   : ")
        if end >= start:
            return start, end
        print("  End must be ≥ start. Try again.")


def ask_pipelines() -> list:
    """
    Ask which pipelines the user wants to run.
    Returns a sorted list of chosen pipeline keys (ints).
    """

    print()
    print("Select which pipeline(s) to run.")
    print("  Enter one or more numbers separated by spaces, or press Enter for all.")
    print()

    for key, cfg in PIPELINE_CONFIGS.items():
        print(f"    [{key}]  {cfg['label']}")

    print()

    while True:
        raw = input("  Your choice (e.g.  1  or  1 3  or  1 2 3): ").strip()

        # Empty → all three
        if raw == "":
            chosen = sorted(PIPELINE_CONFIGS.keys())
            print(f"  → Running all pipelines: {chosen}")
            return chosen

        parts = raw.split()
        try:
            chosen = sorted({int(p) for p in parts})
        except ValueError:
            print("  Invalid input. Enter numbers like: 1  or  2 3  or  1 2 3")
            continue

        invalid = [k for k in chosen if k not in PIPELINE_CONFIGS]
        if invalid:
            print(f"  Unknown pipeline(s): {invalid}. Valid choices are 1, 2, 3.")
            continue

        return chosen


def load_pipeline_module(pipeline_key: int):
    """
    Dynamically loads the pipeline's main module.
    Returns the module object.
    """

    cfg         = PIPELINE_CONFIGS[pipeline_key]
    main_file   = cfg["main_file"]
    module_name = cfg["module_name"]

    if not os.path.isfile(main_file):
        raise FileNotFoundError(
            f"Pipeline main file not found:\n  {main_file}"
        )

    # If already loaded (e.g. user runs the script twice in the same
    # interpreter session) return the cached module.
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

def run_pipeline_for_range(
    pipeline_key: int,
    start: int,
    end: int,
    pipeline_module,
):
    """
    Runs a single pipeline over the requested patient range.
    Uses the pipeline module's own build_config() and run_pipeline()
    so every path and method tag is resolved correctly.
    """

    label = PIPELINE_CONFIGS[pipeline_key]["label"]

    patient_numbers = list(range(start, end + 1))
    total           = len(patient_numbers)

    completed = []
    skipped   = []
    failed    = []

    for position, number in enumerate(patient_numbers, start=1):
        patient_id = make_patient_id(number)
        data_dir   = os.path.join(TRAINING_DIR, patient_id)

        if not os.path.isdir(data_dir):
            print(f"\n  [SKIPPED]  {patient_id}  — data folder not found: {data_dir}")
            skipped.append(patient_id)
            continue

        try:
            config = pipeline_module.build_config(patient_id)

            # Print a compact header so the console stays readable
            print()
            print("-" * 75)
            print(f"  {label}  |  Patient {position}/{total}  |  {patient_id}")
            print("-" * 75)

            pipeline_module.run_pipeline(config)
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

    # --------------------------------------------------------
    # Banner
    # --------------------------------------------------------
    print()
    print("=" * 75)
    print("       BRAIN TUMOR SEGMENTATION  —  UNIFIED LAUNCHER")
    print("=" * 75)
    print()
    print("  Pipelines available:")
    for key, cfg in PIPELINE_CONFIGS.items():
        print(f"    [{key}]  {cfg['label']}")

    # --------------------------------------------------------
    # User inputs
    # --------------------------------------------------------
    start, end = ask_patient_range()
    pipelines  = ask_pipelines()

    # --------------------------------------------------------
    # Confirm before starting
    # --------------------------------------------------------
    print()
    print("=" * 75)
    print("  CONFIGURATION SUMMARY")
    print("=" * 75)
    print(f"  Patient range  :  {make_patient_id(start)}  →  {make_patient_id(end)}")
    print(f"  Total patients :  {end - start + 1}")
    print(f"  Pipelines      :  {[PIPELINE_CONFIGS[k]['label'] for k in pipelines]}")
    print()

    confirm = input("  Press Enter to start, or type 'q' to quit: ").strip().lower()
    if confirm == "q":
        print("  Aborted.")
        return

    # --------------------------------------------------------
    # Load pipeline modules up front so import errors surface early
    # --------------------------------------------------------
    modules = {}
    for key in pipelines:
        print(f"\n  Loading {PIPELINE_CONFIGS[key]['label']} …")
        try:
            modules[key] = load_pipeline_module(key)
            print(f"  OK")
        except Exception as error:
            print(f"  [ERROR] Could not load pipeline {key}: {error}")
            traceback.print_exc()
            return

    # --------------------------------------------------------
    # Run each selected pipeline
    # --------------------------------------------------------
    overall_summary = {}

    for key in pipelines:
        label  = PIPELINE_CONFIGS[key]["label"]
        module = modules[key]

        print()
        print("=" * 75)
        print(f"  STARTING  {label.upper()}")
        print(f"  Range: {make_patient_id(start)}  →  {make_patient_id(end)}")
        print("=" * 75)

        completed, skipped, failed = run_pipeline_for_range(
            key, start, end, module
        )

        overall_summary[key] = {
            "label":     label,
            "completed": completed,
            "skipped":   skipped,
            "failed":    failed,
        }

        print()
        print("=" * 75)
        print(f"  {label.upper()}  —  DONE")
        print(f"  Completed: {len(completed)}   Skipped: {len(skipped)}   Failed: {len(failed)}")
        print("=" * 75)

    # --------------------------------------------------------
    # Grand summary
    # --------------------------------------------------------
    print()
    print("=" * 75)
    print("                      OVERALL BATCH SUMMARY")
    print("=" * 75)
    print(f"  Patient range  :  {make_patient_id(start)}  →  {make_patient_id(end)}")
    print()

    for key, info in overall_summary.items():
        print(f"  {info['label']}")
        print(f"    Completed : {len(info['completed'])}")
        print(f"    Skipped   : {len(info['skipped'])}")
        print(f"    Failed    : {len(info['failed'])}")

        if info["skipped"]:
            print(f"    Skipped   : {', '.join(info['skipped'])}")
        if info["failed"]:
            print(f"    Failed    : {', '.join(info['failed'])}")
        print()

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
