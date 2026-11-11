import os
import sys
import traceback

# ============================================================
# PROJECT CONFIGURATION (PIPELINE 2.0 - WATERSHED)
# ============================================================

PATIENT_PREFIX = "BraTS20_Training_"
PIPELINE_NAME = "Pipeline_2_Watershed"
PIPELINE_VERSION = "v2_watershed"
METHOD_TAG = "watershed"
EVALUATION_REGION = "WT"
PATIENT_NUMBER_WIDTH = 3


# ============================================================
# PROJECT PATHS & MODULE RESOLUTION
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Check if running from inside "Pipeline 2.0" folder or project root
if os.path.basename(_THIS_DIR).lower() in ["pipeline 2.0", "pipeline_2.0"]:
    PROJECT_DIR = os.path.dirname(_THIS_DIR)
    PIPELINE_DIR = _THIS_DIR
else:
    PROJECT_DIR = _THIS_DIR
    PIPELINE_DIR = os.path.join(PROJECT_DIR, "Pipeline 2.0")

CODES_DIR = os.path.join(PROJECT_DIR, "Codes")

for path in [PIPELINE_DIR, CODES_DIR, PROJECT_DIR]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

os.chdir(PROJECT_DIR)


# ============================================================
# VERIFY REQUIRED MODULES
# ============================================================

import preprocessing_II
import segmentation_II
import morphological_analysis_II
import segmentation_evaluation


# ============================================================
# BASE DIRECTORIES
# ============================================================

TRAINING_DIR = os.path.join(PROJECT_DIR, "data", "Training")
OUTPUTS_ROOT = os.path.join(PROJECT_DIR, "Outputs", PIPELINE_NAME)
RESULTS_ROOT = os.path.join(PROJECT_DIR, "Results", PIPELINE_NAME)


# ============================================================
# HELPERS
# ============================================================

def extract_patient_number(patient_id: str) -> str:
    return patient_id.split("_")[-1]


def make_patient_id(number: int) -> str:
    return f"{PATIENT_PREFIX}{number:0{PATIENT_NUMBER_WIDTH}d}"


def build_config(patient_id: str) -> dict:
    patient_number = extract_patient_number(patient_id)
    run_id = f"main_{PIPELINE_VERSION}_{patient_number}"
    data_dir = os.path.join(TRAINING_DIR, patient_id)

    denoised_dir = os.path.join(OUTPUTS_ROOT, "Denoised", patient_id)
    clahe_dir = os.path.join(OUTPUTS_ROOT, "CLAHE", patient_id)
    segmented_dir = os.path.join(OUTPUTS_ROOT, "Segmented", patient_id)

    image_dir = os.path.join(RESULTS_ROOT, "Images", patient_id)
    table_dir = os.path.join(RESULTS_ROOT, "Tables", patient_id)
    graph_dir = os.path.join(RESULTS_ROOT, "Graphs", patient_id)

    for directory in [denoised_dir, clahe_dir, segmented_dir, image_dir, table_dir, graph_dir]:
        os.makedirs(directory, exist_ok=True)

    return {
        "project_dir": PROJECT_DIR,
        "pipeline_dir": PIPELINE_DIR,
        "codes_dir": CODES_DIR,
        "pipeline_name": PIPELINE_NAME,
        "method_tag": METHOD_TAG,
        "patient_id": patient_id,
        "patient_number": patient_number,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "evaluation_region": EVALUATION_REGION,
        "data_dir": data_dir,
        "denoised_dir": denoised_dir,
        "clahe_dir": clahe_dir,
        "segmented_dir": segmented_dir,
        "image_dir": image_dir,
        "table_dir": table_dir,
        "graph_dir": graph_dir
    }


# ============================================================
# USER INPUT
# ============================================================

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


def ask_patient_range() -> tuple:
    print()
    print("=" * 75)
    print("       BRAIN TUMOR SEGMENTATION PIPELINE 2.0 (WATERSHED)")
    print("=" * 75)
    print()
    print("Enter the range of patients to process with Watershed Segmentation.")
    print(f"Example: 1 to 20 -> {make_patient_id(1)} ... {make_patient_id(20)}")
    print()

    start = ask_integer("Start patient number : ")
    while True:
        end = ask_integer("End patient number   : ")
        if end >= start:
            return start, end
        print("  End must be greater than or equal to start.")


# ============================================================
# PIPELINE DISPLAY
# ============================================================

def print_configuration(config: dict, position: int, total: int):
    print()
    print("=" * 75)
    print(f"  PATIENT {position} / {total}  (PIPELINE 2.0 - WATERSHED)")
    print("=" * 75)
    print(f"Patient ID          : {config['patient_id']}")
    print(f"Pipeline            : {config['pipeline_name']} ({config['pipeline_version']})")
    print(f"Algorithm           : 3D Marker-Controlled Watershed")
    print(f"Run ID              : {config['run_id']}")
    print(f"Evaluation Region   : {config['evaluation_region']}")
    print(f"Data directory      : {config['data_dir']}")
    print(f"Outputs directory   : {OUTPUTS_ROOT}")
    print(f"Results directory   : {RESULTS_ROOT}")
    print("=" * 75)


# ============================================================
# SINGLE PATIENT PIPELINE EXECUTION
# ============================================================

def run_pipeline(config: dict):
    patient_id = config["patient_id"]

    # --------------------------------------------------------
    # PHASE 1: PREPROCESSING
    # --------------------------------------------------------
    print()
    print("#" * 75)
    print("# PHASE 1 / 3 : PREPROCESSING (NORMALIZATION & EDGE-PRESERVING FILTERING)")
    print("#" * 75)
    preprocessing_II.run_preprocessing(config)

    # --------------------------------------------------------
    # PHASE 2: WATERSHED SEGMENTATION
    # --------------------------------------------------------
    print()
    print("#" * 75)
    print("# PHASE 2 / 3 : 3D MARKER-CONTROLLED WATERSHED SEGMENTATION")
    print("#" * 75)
    segmentation_output = segmentation_II.run_segmentation(config)

    # --------------------------------------------------------
    # PHASE 3: MORPHOLOGICAL ANALYSIS
    # --------------------------------------------------------
    print()
    print("#" * 75)
    print("# PHASE 3 / 3 : MORPHOLOGICAL & GEOMETRIC ANALYSIS")
    print("#" * 75)
    morphological_analysis_II.analyze_mask(segmentation_output, config)

    # --------------------------------------------------------
    # EVALUATION AGAINST GROUND TRUTH
    # --------------------------------------------------------
    print()
    print("#" * 75)
    print("# EVALUATION : QUANTITATIVE EVALUATION (DICE, IOU, HD95, ASSD)")
    print("#" * 75)
    gt_nii = os.path.join(config["data_dir"], f"{patient_id}_seg.nii")
    gt_gz = os.path.join(config["data_dir"], f"{patient_id}_seg.nii.gz")
    ground_truth_path = gt_nii if os.path.isfile(gt_nii) else gt_gz

    if os.path.isfile(ground_truth_path):
        segmentation_evaluation.evaluate(
            segmentation_output,
            ground_truth_path,
            config["evaluation_region"],
            config
        )
    else:
        print(f"Note: Ground truth file not found at {ground_truth_path}. Skipping evaluation.")

    print()
    print("=" * 75)
    print(f"  PATIENT {patient_id} COMPLETED SUCCESSFULLY (WATERSHED)")
    print("=" * 75)
    print(f"Outputs saved to: {OUTPUTS_ROOT}")
    print(f"Results saved to: {RESULTS_ROOT}")
    print("=" * 75)


# ============================================================
# BATCH EXECUTION
# ============================================================

def main():
    start, end = ask_patient_range()
    patient_numbers = list(range(start, end + 1))
    total = len(patient_numbers)

    completed = []
    skipped = []
    failed = []

    for position, number in enumerate(patient_numbers, start=1):
        patient_id = make_patient_id(number)
        data_dir = os.path.join(TRAINING_DIR, patient_id)

        if not os.path.isdir(data_dir):
            print(f"\n[SKIPPED] {patient_id}: Data folder missing: {data_dir}")
            skipped.append(patient_id)
            continue

        try:
            config = build_config(patient_id)
            print_configuration(config, position, total)
            run_pipeline(config)
            completed.append(patient_id)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping.")
            break
        except Exception as error:
            print(f"\n[FAILED] {patient_id}: {type(error).__name__}: {error}")
            traceback.print_exc()
            failed.append(patient_id)

    print()
    print("=" * 75)
    print("                     BATCH SUMMARY (PIPELINE 2.0)")
    print("=" * 75)
    print(f"Range requested : {make_patient_id(start)} -> {make_patient_id(end)}")
    print(f"Total patients  : {total}")
    print(f"Completed       : {len(completed)}")
    print(f"Skipped         : {len(skipped)}")
    print(f"Failed          : {len(failed)}")

    if skipped:
        print("\nSkipped (data missing):", ", ".join(skipped))
    if failed:
        print("\nFailed:", ", ".join(failed))
    print("=" * 75)


if __name__ == "__main__":
    main()
