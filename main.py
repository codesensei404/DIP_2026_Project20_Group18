import os
import sys


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PATIENT_ID = "BraTS20_Training_002"

# Change this whenever the segmentation/preprocessing approach
# itself changes.
PIPELINE_VERSION = "v1"

# BraTS evaluation region
# WT = Whole Tumor
# TC = Tumor Core
# ET = Enhancing Tumor
EVALUATION_REGION = "WT"


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = BASE_DIR

CODES_DIR = os.path.join(
    PROJECT_DIR,
    "Codes"
)

REQUIRED_MODULES = [
    "preprocessing",
    "segmentation",
    "morphological_analysis",
    "segmentation_evaluation"
]

# Make sure the Codes folder exists and contains every module
if not os.path.isdir(CODES_DIR):
    raise FileNotFoundError(
        f"Codes folder not found: {CODES_DIR}"
    )

for module_name in REQUIRED_MODULES:
    module_path = os.path.join(
        CODES_DIR,
        f"{module_name}.py"
    )
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"Module file not found: {module_path}"
        )

# Make the modules in Codes importable (project root first, then Codes)
for path in (CODES_DIR, PROJECT_DIR):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

# Always work from the project root, so any relative path used inside
# the modules (e.g. "Outputs/...", "Results/...") resolves to the root
# folders and not to Codes/
os.chdir(PROJECT_DIR)


# ============================================================
# RUN IDENTIFIER
# ============================================================

def extract_patient_number(patient_id):
    """
    Extracts the numeric patient identifier.

    Example:
        BraTS20_Training_001 -> 001
        BraTS20_Training_025 -> 025
    """

    return patient_id.split("_")[-1]


PATIENT_NUMBER = extract_patient_number(
    PATIENT_ID
)

RUN_ID = (
    f"main_{PIPELINE_VERSION}_"
    f"{PATIENT_NUMBER}"
)


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "Data",
    "Training",
    PATIENT_ID
)

OUTPUTS_DIR = os.path.join(
    PROJECT_DIR,
    "Outputs"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "Results"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

DENOISED_DIR = os.path.join(
    OUTPUTS_DIR,
    "Denoised",
    PATIENT_ID
)

CLAHE_DIR = os.path.join(
    OUTPUTS_DIR,
    "CLAHE",
    PATIENT_ID
)

SEGMENTED_DIR = os.path.join(
    OUTPUTS_DIR,
    "Segmented",
    PATIENT_ID
)


# ============================================================
# RESULT DIRECTORIES
# ============================================================

IMAGE_DIR = os.path.join(
    RESULTS_DIR,
    "Images",
    PATIENT_ID
)

TABLE_DIR = os.path.join(
    RESULTS_DIR,
    "Tables",
    PATIENT_ID
)

GRAPH_DIR = os.path.join(
    RESULTS_DIR,
    "Graphs",
    PATIENT_ID
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

ALL_DIRECTORIES = [
    DENOISED_DIR,
    CLAHE_DIR,
    SEGMENTED_DIR,
    IMAGE_DIR,
    TABLE_DIR,
    GRAPH_DIR
]

for directory in ALL_DIRECTORIES:
    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# CONFIGURATION OBJECT
# ============================================================

CONFIG = {

    "project_dir": PROJECT_DIR,

    "codes_dir": CODES_DIR,

    "patient_id": PATIENT_ID,

    "patient_number": PATIENT_NUMBER,

    "pipeline_version": PIPELINE_VERSION,

    "run_id": RUN_ID,

    "evaluation_region": EVALUATION_REGION,

    "data_dir": DATA_DIR,

    "denoised_dir": DENOISED_DIR,

    "clahe_dir": CLAHE_DIR,

    "segmented_dir": SEGMENTED_DIR,

    "image_dir": IMAGE_DIR,

    "table_dir": TABLE_DIR,

    "graph_dir": GRAPH_DIR
}


# ============================================================
# IMPORT PIPELINE MODULES
# ============================================================

import preprocessing # type: ignore
import segmentation # type: ignore
import morphological_analysis # type: ignore
import segmentation_evaluation # type: ignore


# ============================================================
# PIPELINE DISPLAY
# ============================================================

def print_configuration():

    print()
    print("=" * 75)
    print("           BRAIN TUMOR DETECTION PIPELINE")
    print("=" * 75)

    print()
    print(f"Patient ID          : {PATIENT_ID}")
    print(f"Patient number      : {PATIENT_NUMBER}")
    print(f"Pipeline version    : {PIPELINE_VERSION}")
    print(f"Run ID              : {RUN_ID}")
    print(f"Evaluation region   : {EVALUATION_REGION}")

    print()
    print("Project directory:")
    print(PROJECT_DIR)

    print()
    print("Codes directory:")
    print(CODES_DIR)

    print()
    print("Data directory:")
    print(DATA_DIR)

    print()
    print("Run directories:")

    print(
        f"  Denoised          : {DENOISED_DIR}"
    )

    print(
        f"  CLAHE             : {CLAHE_DIR}"
    )

    print(
        f"  Segmented         : {SEGMENTED_DIR}"
    )

    print(
        f"  Images            : {IMAGE_DIR}"
    )

    print(
        f"  Tables            : {TABLE_DIR}"
    )

    print(
        f"  Graphs            : {GRAPH_DIR}"
    )

    print()
    print("=" * 75)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print_configuration()

    # --------------------------------------------------------
    # STEP 1
    # PREPROCESSING
    # --------------------------------------------------------

    print()
    print("#" * 75)
    print("# STEP 1 / 4 : PREPROCESSING")
    print("#" * 75)

    preprocessing.run_preprocessing(
        CONFIG
    )

    # --------------------------------------------------------
    # STEP 2
    # SEGMENTATION
    # --------------------------------------------------------

    print()
    print("#" * 75)
    print("# STEP 2 / 4 : SEGMENTATION")
    print("#" * 75)

    segmentation_output = (
        segmentation.run_segmentation(
            CONFIG
        )
    )

    # --------------------------------------------------------
    # STEP 3
    # MORPHOLOGICAL ANALYSIS
    # --------------------------------------------------------

    print()
    print("#" * 75)
    print("# STEP 3 / 4 : MORPHOLOGICAL ANALYSIS")
    print("#" * 75)

    morphological_analysis.analyze_mask(
        segmentation_output,
        CONFIG
    )

    # --------------------------------------------------------
    # STEP 4
    # SEGMENTATION EVALUATION
    # --------------------------------------------------------

    print()
    print("#" * 75)
    print("# STEP 4 / 4 : SEGMENTATION EVALUATION")
    print("#" * 75)

    ground_truth_path = os.path.join(
        DATA_DIR,
        f"{PATIENT_ID}_seg.nii"
    )

    segmentation_evaluation.evaluate(
        segmentation_output,
        ground_truth_path,
        EVALUATION_REGION,
        CONFIG
    )

    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("             PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print()
    print(f"Patient : {PATIENT_ID}")
    print(f"Run     : {RUN_ID}")

    print()
    print("All outputs for this experiment are stored under:")

    print(
        f"  Outputs/Denoised/{PATIENT_ID}/"
    )

    print(
        f"  Outputs/CLAHE/{PATIENT_ID}/"
    )

    print(
        f"  Outputs/Segmented/{PATIENT_ID}/"
    )

    print(
        f"  Results/Images/{PATIENT_ID}/"
    )

    print(
        f"  Results/Tables/{PATIENT_ID}/"
    )

    print(
        f"  Results/Graphs/{PATIENT_ID}/"
    )

    print()
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()