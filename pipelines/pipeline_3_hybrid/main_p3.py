"""
Pipeline 3.0 — Hybrid Threshold-Guided Watershed Segmentation
Main entry point.

Steps per patient:
  1. Preprocessing     (normalize → bilateral denoise → CLAHE)
  2. Segmentation      (threshold core → constrained watershed flooding)
  3. Morphological     (shape descriptors, surface area, components)
  4. Evaluation        (Dice, IoU, HD95, ASSD vs BraTS ground truth)

Outputs  → Outputs/Pipeline_3_Hybrid/<type>/<patient_id>/
Results  → Results/Pipeline_3_Hybrid/<type>/<patient_id>/
"""

import os
import sys
import traceback

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PIPELINE_NAME        = "Pipeline_3_Hybrid"
PIPELINE_VERSION     = "v3_tuned"
METHOD_TAG           = "hybrid"
EVALUATION_REGION    = "WT"
PATIENT_PREFIX       = "BraTS20_Training_"
PATIENT_NUMBER_WIDTH = 3

# ============================================================
# PATH RESOLUTION
# ============================================================

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))   # up two levels

PIPELINE_DIR = _THIS_DIR
SHARED_DIR   = os.path.join(PROJECT_DIR, "shared")

for _p in [PIPELINE_DIR, SHARED_DIR, PROJECT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(PROJECT_DIR)

# ============================================================
# BASE DIRECTORIES
# ============================================================

TRAINING_DIR  = os.path.join(PROJECT_DIR, "data",    "Training")
OUTPUTS_ROOT  = os.path.join(PROJECT_DIR, "Outputs", PIPELINE_NAME)
RESULTS_ROOT  = os.path.join(PROJECT_DIR, "Results", PIPELINE_NAME)

# ============================================================
# IMPORTS
# ============================================================

import preprocessing_III          as preprocessing
import segmentation_III           as segmentation
import morphological_analysis_III as morphological_analysis
import segmentation_evaluation

# ============================================================
# HELPERS
# ============================================================

def extract_patient_number(patient_id: str) -> str:
    return patient_id.split("_")[-1]


def make_patient_id(number: int) -> str:
    return f"{PATIENT_PREFIX}{number:0{PATIENT_NUMBER_WIDTH}d}"


def build_config(patient_id: str) -> dict:
    patient_number = extract_patient_number(patient_id)
    run_id         = f"p3_{PIPELINE_VERSION}_{patient_number}"

    denoised_dir  = os.path.join(OUTPUTS_ROOT, "Denoised",  patient_id)
    clahe_dir     = os.path.join(OUTPUTS_ROOT, "CLAHE",     patient_id)
    segmented_dir = os.path.join(OUTPUTS_ROOT, "Segmented", patient_id)
    image_dir     = os.path.join(RESULTS_ROOT, "Images",    patient_id)
    table_dir     = os.path.join(RESULTS_ROOT, "Tables",    patient_id)
    graph_dir     = os.path.join(RESULTS_ROOT, "Graphs",    patient_id)

    for d in [denoised_dir, clahe_dir, segmented_dir, image_dir, table_dir, graph_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "project_dir":      PROJECT_DIR,
        "pipeline_dir":     PIPELINE_DIR,
        "shared_dir":       SHARED_DIR,
        "pipeline_name":    PIPELINE_NAME,
        "method_tag":       METHOD_TAG,
        "patient_id":       patient_id,
        "patient_number":   patient_number,
        "pipeline_version": PIPELINE_VERSION,
        "run_id":           run_id,
        "evaluation_region":EVALUATION_REGION,
        "data_dir":         os.path.join(TRAINING_DIR, patient_id),
        "denoised_dir":     denoised_dir,
        "clahe_dir":        clahe_dir,
        "segmented_dir":    segmented_dir,
        "image_dir":        image_dir,
        "table_dir":        table_dir,
        "graph_dir":        graph_dir,
    }


def ask_integer(prompt: str) -> int:
    while True:
        try:
            v = int(input(prompt).strip())
            if v >= 1:
                return v
            print("  Please enter a number >= 1.")
        except ValueError:
            print("  Please enter a whole number.")


def ask_patient_range() -> tuple:
    print()
    print("=" * 75)
    print("  PIPELINE 3.0 (HYBRID)  —  STANDALONE RUN")
    print("=" * 75)
    print()
    start = ask_integer("  Start patient number : ")
    while True:
        end = ask_integer("  End patient number   : ")
        if end >= start:
            return start, end
        print("  End must be >= start.")


def print_configuration(config: dict, position: int, total: int):
    print()
    print("=" * 75)
    print(f"  PATIENT {position}/{total}  —  {PIPELINE_NAME}")
    print("=" * 75)
    print(f"  Patient    : {config['patient_id']}")
    print(f"  Run ID     : {config['run_id']}")
    print(f"  Data       : {config['data_dir']}")
    print(f"  Outputs    : {OUTPUTS_ROOT}")
    print(f"  Results    : {RESULTS_ROOT}")
    print("=" * 75)


# ============================================================
# SINGLE PATIENT PIPELINE
# ============================================================

def run_pipeline(config: dict):
    patient_id = config["patient_id"]

    print()
    print("#" * 75)
    print("# PHASE 1 / 3 : PREPROCESSING (BILATERAL DENOISING + CLAHE)")
    print("#" * 75)
    preprocessing.run_preprocessing(config)

    print()
    print("#" * 75)
    print("# PHASE 2 / 3 : HYBRID THRESHOLD-WATERSHED SEGMENTATION")
    print("#" * 75)
    seg_output = segmentation.run_segmentation(config)

    print()
    print("#" * 75)
    print("# PHASE 3 / 3 : MORPHOLOGICAL & GEOMETRIC ANALYSIS")
    print("#" * 75)
    morphological_analysis.analyze_mask(seg_output, config)

    print()
    print("#" * 75)
    print("# EVALUATION  : QUANTITATIVE EVALUATION (DICE, IOU, HD95, ASSD)")
    print("#" * 75)
    gt_nii  = os.path.join(config["data_dir"], f"{patient_id}_seg.nii")
    gt_gz   = os.path.join(config["data_dir"], f"{patient_id}_seg.nii.gz")
    gt_path = gt_nii if os.path.isfile(gt_nii) else gt_gz

    if os.path.isfile(gt_path):
        segmentation_evaluation.evaluate(
            seg_output, gt_path, config["evaluation_region"], config
        )
    else:
        print(f"  Note: ground truth not found at {gt_path}. Skipping evaluation.")

    print()
    print("=" * 75)
    print(f"  {patient_id}  COMPLETED  ({PIPELINE_NAME})")
    print("=" * 75)


# ============================================================
# BATCH EXECUTION
# ============================================================

def main():
    start, end = ask_patient_range()
    total = end - start + 1

    completed, skipped, failed = [], [], []

    for position, number in enumerate(range(start, end + 1), start=1):
        patient_id = make_patient_id(number)
        data_dir   = os.path.join(TRAINING_DIR, patient_id)

        if not os.path.isdir(data_dir):
            print(f"\n  [SKIPPED] {patient_id}: data folder missing")
            skipped.append(patient_id)
            continue

        try:
            config = build_config(patient_id)
            print_configuration(config, position, total)
            run_pipeline(config)
            completed.append(patient_id)
        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
            break
        except Exception as e:
            print(f"\n  [FAILED] {patient_id}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(patient_id)

    print()
    print("=" * 75)
    print(f"  BATCH SUMMARY  —  {PIPELINE_NAME}")
    print("=" * 75)
    print(f"  Range     : {make_patient_id(start)} → {make_patient_id(end)}")
    print(f"  Completed : {len(completed)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Failed    : {len(failed)}")
    if skipped: print(f"  Skipped list : {', '.join(skipped)}")
    if failed:  print(f"  Failed list  : {', '.join(failed)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
