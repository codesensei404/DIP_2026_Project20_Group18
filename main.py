import os
import sys
import traceback


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PATIENT_PREFIX = "BraTS20_Training_"

# Change this whenever the segmentation/preprocessing approach
# itself changes.
PIPELINE_VERSION = "v1"

# BraTS evaluation region
# WT = Whole Tumor
# TC = Tumor Core
# ET = Enhancing Tumor
EVALUATION_REGION = "WT"

# Number of digits used in the patient number (001, 002, ...)
PATIENT_NUMBER_WIDTH = 3


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
# TOP LEVEL DIRECTORIES
# ============================================================

TRAINING_DIR = os.path.join(
    PROJECT_DIR,
    "Data",
    "Training"
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
# IMPORT PIPELINE MODULES
# ============================================================

import preprocessing
import segmentation
import morphological_analysis
import segmentation_evaluation


# ============================================================
# HELPERS
# ============================================================

def extract_patient_number(patient_id):
    """
    Extracts the numeric patient identifier.

    Example:
        BraTS20_Training_001 -> 001
        BraTS20_Training_025 -> 025
    """

    return patient_id.split("_")[-1]


def make_patient_id(number):
    """
    Builds the patient ID from a number.

    Example:
        1  -> BraTS20_Training_001
        25 -> BraTS20_Training_025
    """

    return f"{PATIENT_PREFIX}{number:0{PATIENT_NUMBER_WIDTH}d}"


def build_config(patient_id):
    """
    Builds the configuration dictionary (paths, run id, etc.)
    for a single patient and creates its output/result folders.
    """

    patient_number = extract_patient_number(
        patient_id
    )

    run_id = (
        f"main_{PIPELINE_VERSION}_"
        f"{patient_number}"
    )

    data_dir = os.path.join(
        TRAINING_DIR,
        patient_id
    )

    denoised_dir = os.path.join(
        OUTPUTS_DIR,
        "Denoised",
        patient_id
    )

    clahe_dir = os.path.join(
        OUTPUTS_DIR,
        "CLAHE",
        patient_id
    )

    segmented_dir = os.path.join(
        OUTPUTS_DIR,
        "Segmented",
        patient_id
    )

    image_dir = os.path.join(
        RESULTS_DIR,
        "Images",
        patient_id
    )

    table_dir = os.path.join(
        RESULTS_DIR,
        "Tables",
        patient_id
    )

    graph_dir = os.path.join(
        RESULTS_DIR,
        "Graphs",
        patient_id
    )

    for directory in [
        denoised_dir,
        clahe_dir,
        segmented_dir,
        image_dir,
        table_dir,
        graph_dir
    ]:
        os.makedirs(
            directory,
            exist_ok=True
        )

    return {

        "project_dir": PROJECT_DIR,

        "codes_dir": CODES_DIR,

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


def ask_patient_range():
    """
    Asks the user for the first and last patient number.

    Example:
        Start : 1
        End   : 20
        -> BraTS20_Training_001 ... BraTS20_Training_020
    """

    print()
    print("=" * 75)
    print("           BRAIN TUMOR DETECTION PIPELINE")
    print("=" * 75)
    print()
    print("Enter the range of patients to process.")
    print(f"Example: 1 to 20 -> {make_patient_id(1)} ... {make_patient_id(20)}")
    print()

    start = ask_integer("Start patient number : ")

    while True:

        end = ask_integer("End patient number   : ")

        if end < start:
            print("  End must be greater than or equal to start.")
            continue

        break

    return start, end


# ============================================================
# PIPELINE DISPLAY
# ============================================================

def print_configuration(config, position, total):

    print()
    print("=" * 75)
    print(f"  PATIENT {position} / {total}")
    print("=" * 75)

    print()
    print(f"Patient ID          : {config['patient_id']}")
    print(f"Patient number      : {config['patient_number']}")
    print(f"Pipeline version    : {config['pipeline_version']}")
    print(f"Run ID              : {config['run_id']}")
    print(f"Evaluation region   : {config['evaluation_region']}")

    print()
    print("Project directory:")
    print(config["project_dir"])

    print()
    print("Codes directory:")
    print(config["codes_dir"])

    print()
    print("Data directory:")
    print(config["data_dir"])

    print()
    print("Run directories:")

    print(
        f"  Denoised          : {config['denoised_dir']}"
    )

    print(
        f"  CLAHE             : {config['clahe_dir']}"
    )

    print(
        f"  Segmented         : {config['segmented_dir']}"
    )

    print(
        f"  Images            : {config['image_dir']}"
    )

    print(
        f"  Tables            : {config['table_dir']}"
    )

    print(
        f"  Graphs            : {config['graph_dir']}"
    )

    print()
    print("=" * 75)


# ============================================================
# SINGLE PATIENT PIPELINE
# ============================================================

def run_pipeline(config):

    patient_id = config["patient_id"]
    run_id = config["run_id"]

    # --------------------------------------------------------
    # STEP 1
    # PREPROCESSING
    # --------------------------------------------------------

    print()
    print("#" * 75)
    print("# STEP 1 / 4 : PREPROCESSING")
    print("#" * 75)

    preprocessing.run_preprocessing(
        config
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
            config
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
        config
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
        config["data_dir"],
        f"{patient_id}_seg.nii"
    )

    segmentation_evaluation.evaluate(
        segmentation_output,
        ground_truth_path,
        config["evaluation_region"],
        config
    )

    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("             PATIENT COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print()
    print(f"Patient : {patient_id}")
    print(f"Run     : {run_id}")

    print()
    print("All outputs for this experiment are stored under:")

    print(
        f"  Outputs/Denoised/{patient_id}/"
    )

    print(
        f"  Outputs/CLAHE/{patient_id}/"
    )

    print(
        f"  Outputs/Segmented/{patient_id}/"
    )

    print(
        f"  Results/Images/{patient_id}/"
    )

    print(
        f"  Results/Tables/{patient_id}/"
    )

    print(
        f"  Results/Graphs/{patient_id}/"
    )

    print()
    print("=" * 75)


# ============================================================
# MAIN PIPELINE (BATCH)
# ============================================================

def main():

    start, end = ask_patient_range()

    patient_numbers = list(
        range(start, end + 1)
    )

    total = len(patient_numbers)

    completed = []
    skipped = []
    failed = []

    for position, number in enumerate(
        patient_numbers,
        start=1
    ):

        patient_id = make_patient_id(number)

        data_dir = os.path.join(
            TRAINING_DIR,
            patient_id
        )

        # Skip patients whose data folder does not exist
        if not os.path.isdir(data_dir):

            print()
            print(
                f"[SKIPPED] {patient_id}: "
                f"data folder not found: {data_dir}"
            )

            skipped.append(patient_id)
            continue

        try:

            config = build_config(
                patient_id
            )

            print_configuration(
                config,
                position,
                total
            )

            run_pipeline(
                config
            )

            completed.append(patient_id)

        except KeyboardInterrupt:

            print()
            print("Interrupted by user. Stopping.")
            break

        except Exception as error:

            print()
            print(
                f"[FAILED] {patient_id}: "
                f"{type(error).__name__}: {error}"
            )

            traceback.print_exc()

            failed.append(patient_id)

    # --------------------------------------------------------
    # BATCH SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("                     BATCH SUMMARY")
    print("=" * 75)

    print()
    print(f"Range requested : {make_patient_id(start)} -> {make_patient_id(end)}")
    print(f"Total patients  : {total}")
    print(f"Completed       : {len(completed)}")
    print(f"Skipped         : {len(skipped)}")
    print(f"Failed          : {len(failed)}")

    if skipped:
        print()
        print("Skipped (data folder missing):")
        for patient_id in skipped:
            print(f"  {patient_id}")

    if failed:
        print()
        print("Failed (see error messages above):")
        for patient_id in failed:
            print(f"  {patient_id}")

    print()
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()