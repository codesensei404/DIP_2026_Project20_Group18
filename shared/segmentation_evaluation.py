import os
import json
import csv

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"NIfTI file not found:\n"
            f"{path}"
        )

    image = sitk.ReadImage(
        path
    )

    array = sitk.GetArrayFromImage(
        image
    )

    return array, image


# ============================================================
# BRA TS REGION CONVERSION
# ============================================================

def convert_ground_truth(
    ground_truth,
    region="WT"
):

    region = region.upper()

    if region == "WT":

        mask = (
            (ground_truth == 1) |
            (ground_truth == 2) |
            (ground_truth == 4)
        )

    elif region == "TC":

        mask = (
            (ground_truth == 1) |
            (ground_truth == 4)
        )

    elif region == "ET":

        mask = (
            ground_truth == 4
        )

    else:

        raise ValueError(
            "Region must be WT, TC, or ET."
        )

    return mask


# ============================================================
# MASK PREPARATION
# ============================================================

def prepare_prediction(
    prediction
):

    return prediction > 0


# ============================================================
# COMPATIBILITY
# ============================================================

def verify_compatibility(
    prediction_image,
    ground_truth_image
):

    if (
        prediction_image.GetSize()
        !=
        ground_truth_image.GetSize()
    ):

        raise ValueError(
            "Prediction and ground truth "
            "have different sizes."
        )

    if not np.allclose(
        prediction_image.GetSpacing(),
        ground_truth_image.GetSpacing()
    ):

        raise ValueError(
            "Prediction and ground truth "
            "have different voxel spacing."
        )

    if not np.allclose(
        prediction_image.GetOrigin(),
        ground_truth_image.GetOrigin()
    ):

        raise ValueError(
            "Prediction and ground truth "
            "have different origins."
        )

    if not np.allclose(
        prediction_image.GetDirection(),
        ground_truth_image.GetDirection()
    ):

        raise ValueError(
            "Prediction and ground truth "
            "have different directions."
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    prediction,
    ground_truth
):

    tp = np.sum(
        prediction & ground_truth
    )

    tn = np.sum(
        (~prediction) & (~ground_truth)
    )

    fp = np.sum(
        prediction & (~ground_truth)
    )

    fn = np.sum(
        (~prediction) & ground_truth
    )

    return (
        int(tp),
        int(tn),
        int(fp),
        int(fn)
    )


# ============================================================
# DICE AND IOU
# ============================================================

def calculate_overlap_metrics(
    tp,
    fp,
    fn
):

    denominator_dice = (
        2 * tp +
        fp +
        fn
    )

    denominator_iou = (
        tp +
        fp +
        fn
    )

    if denominator_dice == 0:

        dice = 1.0

    else:

        dice = (
            2 * tp /
            denominator_dice
        )

    if denominator_iou == 0:

        iou = 1.0

    else:

        iou = (
            tp /
            denominator_iou
        )

    return dice, iou


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_classification_metrics(
    tp,
    tn,
    fp,
    fn
):

    precision = (
        tp /
        (tp + fp)
        if (tp + fp) > 0
        else 1.0
    )

    sensitivity = (
        tp /
        (tp + fn)
        if (tp + fn) > 0
        else 1.0
    )

    specificity = (
        tn /
        (tn + fp)
        if (tn + fp) > 0
        else 1.0
    )

    false_positive_rate = (
        fp /
        (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn /
        (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn)
        if (tp + tn + fp + fn) > 0
        else 1.0
    )

    return {

        "precision":
            float(precision),

        "sensitivity_recall":
            float(sensitivity),

        "specificity":
            float(specificity),

        "false_positive_rate":
            float(false_positive_rate),

        "false_negative_rate":
            float(false_negative_rate),

        "voxel_accuracy":
            float(accuracy)
    }


# ============================================================
# VOLUME METRICS
# ============================================================

def calculate_volume_metrics(
    prediction,
    ground_truth,
    spacing
):

    voxel_volume_mm3 = np.prod(
        spacing
    )

    predicted_voxels = int(
        np.sum(prediction)
    )

    ground_truth_voxels = int(
        np.sum(ground_truth)
    )

    predicted_volume = (
        predicted_voxels *
        voxel_volume_mm3
    )

    ground_truth_volume = (
        ground_truth_voxels *
        voxel_volume_mm3
    )

    absolute_difference = abs(
        predicted_volume -
        ground_truth_volume
    )

    if ground_truth_volume > 0:

        relative_error = (
            absolute_difference /
            ground_truth_volume
        )

        denominator = (
            predicted_volume +
            ground_truth_volume
        )

        if denominator > 0:

            volume_similarity = (
                1.0 -
                absolute_difference /
                denominator
            )

        else:

            volume_similarity = 1.0

    else:

        relative_error = 0.0
        volume_similarity = 1.0

    return {

        "predicted_voxel_count":
            predicted_voxels,

        "ground_truth_voxel_count":
            ground_truth_voxels,

        "predicted_volume_mm3":
            float(predicted_volume),

        "ground_truth_volume_mm3":
            float(ground_truth_volume),

        "predicted_volume_cm3":
            float(
                predicted_volume / 1000.0
            ),

        "ground_truth_volume_cm3":
            float(
                ground_truth_volume / 1000.0
            ),

        "absolute_volume_difference_mm3":
            float(absolute_difference),

        "relative_volume_error":
            float(relative_error),

        "volume_similarity":
            float(volume_similarity)
    }


# ============================================================
# BOUNDING BOX
# ============================================================

def get_bounding_box(
    mask
):

    coordinates = np.argwhere(
        mask
    )

    if len(coordinates) == 0:

        return None

    z_min, y_min, x_min = (
        coordinates.min(axis=0)
    )

    z_max, y_max, x_max = (
        coordinates.max(axis=0)
    )

    return {

        "z_min": int(z_min),
        "z_max": int(z_max),

        "y_min": int(y_min),
        "y_max": int(y_max),

        "x_min": int(x_min),
        "x_max": int(x_max)
    }


def bounding_box_iou(
    prediction,
    ground_truth
):

    pred_box = get_bounding_box(
        prediction
    )

    gt_box = get_bounding_box(
        ground_truth
    )

    if (
        pred_box is None
        and
        gt_box is None
    ):

        return 1.0

    if (
        pred_box is None
        or
        gt_box is None
    ):

        return 0.0

    z_min = max(
        pred_box["z_min"],
        gt_box["z_min"]
    )

    z_max = min(
        pred_box["z_max"],
        gt_box["z_max"]
    )

    y_min = max(
        pred_box["y_min"],
        gt_box["y_min"]
    )

    y_max = min(
        pred_box["y_max"],
        gt_box["y_max"]
    )

    x_min = max(
        pred_box["x_min"],
        gt_box["x_min"]
    )

    x_max = min(
        pred_box["x_max"],
        gt_box["x_max"]
    )

    if (
        z_max < z_min
        or
        y_max < y_min
        or
        x_max < x_min
    ):

        intersection = 0

    else:

        intersection = (
            (z_max - z_min + 1) *
            (y_max - y_min + 1) *
            (x_max - x_min + 1)
        )

    pred_volume = (
        (pred_box["z_max"] -
         pred_box["z_min"] + 1) *
        (pred_box["y_max"] -
         pred_box["y_min"] + 1) *
        (pred_box["x_max"] -
         pred_box["x_min"] + 1)
    )

    gt_volume = (
        (gt_box["z_max"] -
         gt_box["z_min"] + 1) *
        (gt_box["y_max"] -
         gt_box["y_min"] + 1) *
        (gt_box["x_max"] -
         gt_box["x_min"] + 1)
    )

    union = (
        pred_volume +
        gt_volume -
        intersection
    )

    if union == 0:

        return 1.0

    return (
        intersection /
        union
    )


# ============================================================
# SURFACE EXTRACTION
# ============================================================

def extract_surface(
    mask
):

    structure = (
        ndimage.generate_binary_structure(
            3,
            1
        )
    )

    eroded = ndimage.binary_erosion(
        mask,
        structure=structure
    )

    surface = (
        mask ^ eroded
    )

    return surface


# ============================================================
# HD95 + ASSD
# ============================================================

def calculate_surface_distances(
    prediction,
    ground_truth,
    spacing
):

    pred_surface = extract_surface(
        prediction
    )

    gt_surface = extract_surface(
        ground_truth
    )

    if (
        np.sum(pred_surface) == 0
        or
        np.sum(gt_surface) == 0
    ):

        return {
            "hd95_mm": None,
            "assd_mm": None
        }

    gt_distance = (
        ndimage.distance_transform_edt(
            ~gt_surface,
            sampling=(
                spacing[2],
                spacing[1],
                spacing[0]
            )
        )
    )

    pred_distance = (
        ndimage.distance_transform_edt(
            ~pred_surface,
            sampling=(
                spacing[2],
                spacing[1],
                spacing[0]
            )
        )
    )

    pred_to_gt = (
        gt_distance[
            pred_surface
        ]
    )

    gt_to_pred = (
        pred_distance[
            gt_surface
        ]
    )

    all_distances = np.concatenate([
        pred_to_gt,
        gt_to_pred
    ])

    hd95 = np.percentile(
        all_distances,
        95
    )

    assd = np.mean(
        all_distances
    )

    return {

        "hd95_mm":
            float(hd95),

        "assd_mm":
            float(assd)
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    prediction_path,
    ground_truth_path,
    region,
    config
):

    patient_id = config[
        "patient_id"
    ]

    run_id = config[
        "run_id"
    ]

    table_dir = config[
        "table_dir"
    ]

    print()
    print("=" * 65)
    print("SEGMENTATION EVALUATION")
    print("=" * 65)

    print()
    print("Prediction:")
    print(prediction_path)

    print()
    print("Ground Truth:")
    print(ground_truth_path)

    print()
    print("BraTS region:", region)

    prediction_array, prediction_image = (
        load_nifti(
            prediction_path
        )
    )

    ground_truth_array, ground_truth_image = (
        load_nifti(
            ground_truth_path
        )
    )

    verify_compatibility(
        prediction_image,
        ground_truth_image
    )

    prediction = prepare_prediction(
        prediction_array
    )

    ground_truth = convert_ground_truth(
        ground_truth_array,
        region
    )

    spacing = prediction_image.GetSpacing()

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp, tn, fp, fn = (
        calculate_confusion_matrix(
            prediction,
            ground_truth
        )
    )

    # --------------------------------------------------------
    # Dice + IoU
    # --------------------------------------------------------

    dice, iou = (
        calculate_overlap_metrics(
            tp,
            fp,
            fn
        )
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification = (
        calculate_classification_metrics(
            tp,
            tn,
            fp,
            fn
        )
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume = (
        calculate_volume_metrics(
            prediction,
            ground_truth,
            spacing
        )
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bbox_iou = (
        bounding_box_iou(
            prediction,
            ground_truth
        )
    )

    # --------------------------------------------------------
    # Surface
    # --------------------------------------------------------

    surface = (
        calculate_surface_distances(
            prediction,
            ground_truth,
            spacing
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "patient_id":
            patient_id,

        "run_id":
            run_id,

        "region":
            region,

        "true_positive_voxels":
            tp,

        "true_negative_voxels":
            tn,

        "false_positive_voxels":
            fp,

        "false_negative_voxels":
            fn,

        "dice":
            float(dice),

        "dice_percentage":
            float(dice * 100.0),

        "iou_jaccard":
            float(iou),

        "iou_percentage":
            float(iou * 100.0),

        "bounding_box_iou":
            float(bbox_iou),

        **classification,

        **volume,

        **surface
    }

    # --------------------------------------------------------
    # Output paths (Supports both method-tagged and standard names)
    # --------------------------------------------------------

    method_tag = config.get("method_tag", "")
    if not method_tag:
        if "watershed" in prediction_path.lower():
            method_tag = "watershed"
        elif "threshold" in prediction_path.lower() or "v1" in str(config.get("pipeline_version", "")):
            method_tag = "thresholding"

    output_csv = os.path.join(
        table_dir,
        f"{patient_id}_{region}_evaluation.csv"
    )

    output_json = os.path.join(
        table_dir,
        f"{patient_id}_{region}_evaluation.json"
    )

    tagged_csv = (
        os.path.join(table_dir, f"{patient_id}_{region}_evaluation_{method_tag}.csv")
        if method_tag else None
    )
    tagged_json = (
        os.path.join(table_dir, f"{patient_id}_{region}_evaluation_{method_tag}.json")
        if method_tag else None
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(
        output_csv,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Metric",
            "Value"
        ])

        for key, value in results.items():

            writer.writerow([
                key,
                value
            ])

    if tagged_csv and tagged_csv != output_csv:
        with open(tagged_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for key, value in results.items():
                writer.writerow([key, value])

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    with open(
        output_json,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    if tagged_json and tagged_json != output_json:
        with open(tagged_json, "w") as file:
            json.dump(results, file, indent=4)

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("-" * 65)
    print("OVERLAP")
    print("-" * 65)

    print(
        f"Dice Similarity Coefficient : "
        f"{dice:.4f} "
        f"({dice * 100:.2f}%)"
    )

    print(
        f"IoU / Jaccard              : "
        f"{iou:.4f} "
        f"({iou * 100:.2f}%)"
    )

    print(
        f"Bounding Box IoU            : "
        f"{bbox_iou:.4f}"
    )

    print()
    print("-" * 65)
    print("CONFUSION MATRIX")
    print("-" * 65)

    print(f"TP : {tp}")
    print(f"TN : {tn}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")

    print()
    print("-" * 65)
    print("CLASSIFICATION METRICS")
    print("-" * 65)

    print(
        f"Precision                   : "
        f"{classification['precision']:.4f}"
    )

    print(
        f"Sensitivity / Recall       : "
        f"{classification['sensitivity_recall']:.4f}"
    )

    print(
        f"Specificity                 : "
        f"{classification['specificity']:.4f}"
    )

    print(
        f"False Positive Rate         : "
        f"{classification['false_positive_rate']:.4f}"
    )

    print(
        f"False Negative Rate         : "
        f"{classification['false_negative_rate']:.4f}"
    )

    print()
    print("-" * 65)
    print("VOLUME AGREEMENT")
    print("-" * 65)

    print(
        f"Predicted volume            : "
        f"{volume['predicted_volume_cm3']:.3f} cm³"
    )

    print(
        f"Ground-truth volume         : "
        f"{volume['ground_truth_volume_cm3']:.3f} cm³"
    )

    print(
        f"Absolute volume difference  : "
        f"{volume['absolute_volume_difference_mm3']:.3f} mm³"
    )

    print(
        f"Relative volume error       : "
        f"{volume['relative_volume_error']:.4f}"
    )

    print(
        f"Volume similarity           : "
        f"{volume['volume_similarity']:.4f}"
    )

    print()
    print("-" * 65)
    print("BOUNDARY AGREEMENT")
    print("-" * 65)

    print(
        f"HD95                        : "
        f"{surface['hd95_mm']} mm"
    )

    print(
        f"ASSD                        : "
        f"{surface['assd_mm']} mm"
    )

    print()
    print("=" * 65)

    print(
        "Results saved to:"
    )

    print(output_csv)
    print(output_json)

    print("=" * 65)

    return results

# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(_THIS_DIR)

    print()
    print("=" * 75)
    print("       BRAIN TUMOR SEGMENTATION EVALUATION (STANDALONE)")
    print("=" * 75)

    default_patient = "BraTS20_Training_001"
    user_input = input(f"Enter patient ID (default: {default_patient}): ").strip()
    patient_id = user_input if user_input else default_patient

    print("Choose segmentation mask to evaluate:")
    print("  [1] Pipeline 1.0 (Thresholding)")
    print("  [2] Pipeline 2.0 (Watershed)")
    choice = input("Choice (1/2, default: 1): ").strip()

    if choice == "2":
        pred_path = os.path.join(PROJECT_DIR, "Outputs", "Pipeline_2_Watershed", "Segmented", patient_id, f"{patient_id}_segmentation_watershed.nii.gz")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(PROJECT_DIR, "Outputs", "Pipeline_2_Watershed", "Segmented", patient_id, f"{patient_id}_segmented.nii.gz")
        tbl_dir = os.path.join(PROJECT_DIR, "Results", "Pipeline_2_Watershed", "Tables", patient_id)
        method_tag = "watershed"
    else:
        pred_path = os.path.join(PROJECT_DIR, "Outputs", "Pipeline_1_Thresholding", "Segmented", patient_id, f"{patient_id}_segmentation_thresholding.nii.gz")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(PROJECT_DIR, "Outputs", "Segmented", patient_id, f"{patient_id}_segmented.nii.gz")
        tbl_dir = os.path.join(PROJECT_DIR, "Results", "Pipeline_1_Thresholding", "Tables", patient_id)
        method_tag = "thresholding"

    gt_path = os.path.join(PROJECT_DIR, "data", "Training", patient_id, f"{patient_id}_seg.nii")

    config = {
        "patient_id": patient_id,
        "run_id": f"standalone_eval_{method_tag}_{patient_id}",
        "table_dir": tbl_dir,
        "method_tag": method_tag,
    }

    os.makedirs(tbl_dir, exist_ok=True)
    evaluate(pred_path, gt_path, "WT", config)
    print(f"\nStandalone evaluation complete for {patient_id} ({method_tag})!")
