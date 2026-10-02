import os
import json
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{path}\n\n"
            f"Run preprocessing first."
        )

    image = sitk.ReadImage(
        path
    )

    volume = sitk.GetArrayFromImage(
        image
    ).astype(np.float32)

    return volume, image


# ============================================================
# VERIFY SEQUENCES
# ============================================================

def verify_sequences(images):

    names = list(
        images.keys()
    )

    reference_name = names[0]

    reference_image = images[
        reference_name
    ][1]

    reference_shape = images[
        reference_name
    ][0].shape

    reference_spacing = (
        reference_image.GetSpacing()
    )

    reference_origin = (
        reference_image.GetOrigin()
    )

    reference_direction = (
        reference_image.GetDirection()
    )

    for name in names[1:]:

        volume = images[name][0]

        image = images[name][1]

        if volume.shape != reference_shape:

            raise ValueError(
                f"Shape mismatch:\n"
                f"{reference_name}: {reference_shape}\n"
                f"{name}: {volume.shape}"
            )

        if not np.allclose(
            image.GetSpacing(),
            reference_spacing
        ):

            raise ValueError(
                f"Spacing mismatch between "
                f"{reference_name} and {name}."
            )

        if not np.allclose(
            image.GetOrigin(),
            reference_origin
        ):

            raise ValueError(
                f"Origin mismatch between "
                f"{reference_name} and {name}."
            )

        if not np.allclose(
            image.GetDirection(),
            reference_direction
        ):

            raise ValueError(
                f"Direction mismatch between "
                f"{reference_name} and {name}."
            )

    print(
        "All MRI sequences have matching "
        "dimensions and spatial metadata."
    )


# ============================================================
# CREATE BRAIN MASK
# ============================================================

def create_brain_mask(
    t1,
    t1ce,
    t2,
    flair
):

    brain_mask = (
        (t1 > 0) |
        (t1ce > 0) |
        (t2 > 0) |
        (flair > 0)
    )

    brain_mask = brain_mask.astype(
        bool
    )

    structure = ndi.generate_binary_structure(
        3,
        2
    )

    brain_mask = ndi.binary_closing(
        brain_mask,
        structure=structure,
        iterations=1
    )

    brain_mask = ndi.binary_fill_holes(
        brain_mask
    )

    return brain_mask


# ============================================================
# PERCENTILE THRESHOLD
# ============================================================

def compute_percentile_threshold(
    volume,
    brain_mask,
    percentile
):

    values = volume[
        brain_mask
    ]

    if values.size == 0:

        return 1.0

    return float(
        np.percentile(
            values,
            percentile
        )
    )


# ============================================================
# MULTI-SEQUENCE CANDIDATE GENERATION
# ============================================================

def generate_tumor_candidates(
    t1,
    t1ce,
    t2,
    flair,
    brain_mask
):

    flair_threshold = (
        compute_percentile_threshold(
            flair,
            brain_mask,
            85
        )
    )

    t2_threshold = (
        compute_percentile_threshold(
            t2,
            brain_mask,
            85
        )
    )

    t1ce_threshold = (
        compute_percentile_threshold(
            t1ce,
            brain_mask,
            90
        )
    )

    print()
    print("Thresholds:")

    print(
        f"FLAIR 85th percentile = "
        f"{flair_threshold:.4f}"
    )

    print(
        f"T2 85th percentile    = "
        f"{t2_threshold:.4f}"
    )

    print(
        f"T1ce 90th percentile  = "
        f"{t1ce_threshold:.4f}"
    )

    flair_high = (
        flair >= flair_threshold
    )

    t2_high = (
        t2 >= t2_threshold
    )

    t1ce_high = (
        t1ce >= t1ce_threshold
    )

    flair_t2_candidate = (
        flair_high &
        t2_high
    )

    t1ce_flair_candidate = (
        t1ce_high &
        flair_high
    )

    final_candidate = (
        flair_t2_candidate |
        t1ce_flair_candidate
    )

    final_candidate &= brain_mask

    return (
        final_candidate,
        flair_t2_candidate,
        t1ce_flair_candidate
    )


# ============================================================
# 3D MORPHOLOGICAL REFINEMENT
# ============================================================

def refine_candidate_mask(
    candidate
):

    structure = ndi.generate_binary_structure(
        3,
        2
    )

    refined = ndi.binary_opening(
        candidate,
        structure=structure,
        iterations=1
    )

    refined = ndi.binary_closing(
        refined,
        structure=structure,
        iterations=2
    )

    refined = ndi.binary_fill_holes(
        refined
    )

    return refined


# ============================================================
# 3D CONNECTED COMPONENT FILTERING
# ============================================================

def filter_connected_components(
    mask,
    reference_image,
    min_volume_mm3=200.0
):

    structure = ndi.generate_binary_structure(
        3,
        2
    )

    labels, num_components = ndi.label(
        mask,
        structure=structure
    )

    print()
    print(
        f"Initial connected components: "
        f"{num_components}"
    )

    spacing = reference_image.GetSpacing()

    voxel_volume_mm3 = (
        spacing[0] *
        spacing[1] *
        spacing[2]
    )

    print(
        f"Voxel volume = "
        f"{voxel_volume_mm3:.4f} mm^3"
    )

    clean_mask = np.zeros(
        mask.shape,
        dtype=bool
    )

    kept = 0

    for component_id in range(
        1,
        num_components + 1
    ):

        voxel_count = np.sum(
            labels == component_id
        )

        physical_volume = (
            voxel_count *
            voxel_volume_mm3
        )

        if physical_volume >= min_volume_mm3:

            clean_mask[
                labels == component_id
            ] = True

            kept += 1

    print(
        f"Components retained: {kept}"
    )

    return clean_mask


# ============================================================
# BOUNDING BOX
# ============================================================

def calculate_bounding_box(
    tumor_mask
):

    coordinates = np.where(
        tumor_mask
    )

    if coordinates[0].size == 0:

        return {
            "found": False
        }

    z_min = int(
        coordinates[0].min()
    )

    z_max = int(
        coordinates[0].max()
    )

    y_min = int(
        coordinates[1].min()
    )

    y_max = int(
        coordinates[1].max()
    )

    x_min = int(
        coordinates[2].min()
    )

    x_max = int(
        coordinates[2].max()
    )

    return {
        "found": True,

        "z_min": z_min,
        "z_max": z_max,

        "y_min": y_min,
        "y_max": y_max,

        "x_min": x_min,
        "x_max": x_max
    }


# ============================================================
# TUMOR VOLUME
# ============================================================

def calculate_tumor_volume(
    tumor_mask,
    reference_image
):

    voxel_count = np.sum(
        tumor_mask
    )

    spacing = reference_image.GetSpacing()

    voxel_volume = (
        spacing[0] *
        spacing[1] *
        spacing[2]
    )

    volume_mm3 = (
        voxel_count *
        voxel_volume
    )

    volume_cm3 = (
        volume_mm3 / 1000.0
    )

    return (
        float(volume_mm3),
        float(volume_cm3)
    )


# ============================================================
# SAVE SEGMENTATION
# ============================================================

def save_segmentation(
    tumor_mask,
    reference_image,
    output_path
):

    segmentation_uint8 = (
        tumor_mask.astype(
            np.uint8
        )
    )

    segmentation_image = (
        sitk.GetImageFromArray(
            segmentation_uint8
        )
    )

    segmentation_image.CopyInformation(
        reference_image
    )

    sitk.WriteImage(
        segmentation_image,
        output_path
    )


# ============================================================
# DIAGNOSTIC IMAGE
# ============================================================

def save_segmentation_diagnostic(
    flair,
    t2,
    t1ce,
    tumor_mask,
    output_path,
    patient_id
):

    slice_areas = np.sum(
        tumor_mask,
        axis=(1, 2)
    )

    best_slice = int(
        np.argmax(
            slice_areas
        )
    )

    flair_slice = flair[
        best_slice
    ]

    t2_slice = t2[
        best_slice
    ]

    t1ce_slice = t1ce[
        best_slice
    ]

    mask_slice = tumor_mask[
        best_slice
    ]

    flair_uint8 = np.clip(
        flair_slice * 255.0,
        0,
        255
    ).astype(np.uint8)

    overlay = cv2.cvtColor(
        flair_uint8,
        cv2.COLOR_GRAY2RGB
    )

    mask_uint8 = (
        mask_slice.astype(
            np.uint8
        ) * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 0, 0),
        2
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4)
    )

    axes[0].imshow(
        flair_slice,
        cmap="gray"
    )
    axes[0].set_title("FLAIR")
    axes[0].axis("off")

    axes[1].imshow(
        t2_slice,
        cmap="gray"
    )
    axes[1].set_title("T2")
    axes[1].axis("off")

    axes[2].imshow(
        t1ce_slice,
        cmap="gray"
    )
    axes[2].set_title("T1ce")
    axes[2].axis("off")

    axes[3].imshow(
        mask_slice,
        cmap="gray"
    )
    axes[3].set_title(
        "Predicted Tumor Mask"
    )
    axes[3].axis("off")

    axes[4].imshow(
        overlay
    )
    axes[4].set_title(
        "Tumor Boundary"
    )
    axes[4].axis("off")

    plt.suptitle(
        f"{patient_id} - "
        f"Segmentation - "
        f"Slice {best_slice}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return best_slice


# ============================================================
# PIPELINE FUNCTION
# ============================================================

def run_segmentation(config):

    patient_id = config["patient_id"]

    denoised_dir = config["denoised_dir"]

    segmented_dir = config["segmented_dir"]

    image_dir = config["image_dir"]

    print()
    print("=" * 65)
    print("MULTI-SEQUENCE BRAIN TUMOR SEGMENTATION")
    print("=" * 65)

    print(
        f"Patient: {patient_id}"
    )

    print(
        f"Run: {config['run_id']}"
    )

    # --------------------------------------------------------
    # Input paths
    # --------------------------------------------------------

    t1_path = os.path.join(
        denoised_dir,
        f"{patient_id}_t1_denoised.nii.gz"
    )

    t1ce_path = os.path.join(
        denoised_dir,
        f"{patient_id}_t1ce_denoised.nii.gz"
    )

    t2_path = os.path.join(
        denoised_dir,
        f"{patient_id}_t2_denoised.nii.gz"
    )

    flair_path = os.path.join(
        denoised_dir,
        f"{patient_id}_flair_denoised.nii.gz"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("Loading MRI sequences...")

    t1, t1_image = load_nifti(
        t1_path
    )

    t1ce, t1ce_image = load_nifti(
        t1ce_path
    )

    t2, t2_image = load_nifti(
        t2_path
    )

    flair, flair_image = load_nifti(
        flair_path
    )

    images = {
        "T1": (t1, t1_image),
        "T1ce": (t1ce, t1ce_image),
        "T2": (t2, t2_image),
        "FLAIR": (flair, flair_image)
    }

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_sequences(
        images
    )

    reference_image = flair_image

    # --------------------------------------------------------
    # Brain mask
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("CREATING 3D BRAIN MASK")
    print("=" * 65)

    brain_mask = create_brain_mask(
        t1,
        t1ce,
        t2,
        flair
    )

    print(
        f"Brain voxels: "
        f"{np.sum(brain_mask)}"
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("MULTI-SEQUENCE CANDIDATE GENERATION")
    print("=" * 65)

    (
        candidate,
        flair_t2_candidate,
        t1ce_flair_candidate
    ) = generate_tumor_candidates(
        t1,
        t1ce,
        t2,
        flair,
        brain_mask
    )

    print(
        f"FLAIR + T2 candidate voxels: "
        f"{np.sum(flair_t2_candidate)}"
    )

    print(
        f"T1ce + FLAIR candidate voxels: "
        f"{np.sum(t1ce_flair_candidate)}"
    )

    print(
        f"Combined candidate voxels: "
        f"{np.sum(candidate)}"
    )

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("3D MORPHOLOGICAL REFINEMENT")
    print("=" * 65)

    refined_mask = refine_candidate_mask(
        candidate
    )

    print(
        f"Voxels after morphology: "
        f"{np.sum(refined_mask)}"
    )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("3D CONNECTED COMPONENT ANALYSIS")
    print("=" * 65)

    final_mask = filter_connected_components(
        refined_mask,
        reference_image,
        min_volume_mm3=200.0
    )

    print(
        f"Final tumor voxels: "
        f"{np.sum(final_mask)}"
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bounding_box = calculate_bounding_box(
        final_mask
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    (
        tumor_volume_mm3,
        tumor_volume_cm3
    ) = calculate_tumor_volume(
        final_mask,
        reference_image
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("LOCALIZATION RESULTS")
    print("=" * 65)

    if bounding_box["found"]:

        print(
            f"X range: "
            f"{bounding_box['x_min']} - "
            f"{bounding_box['x_max']}"
        )

        print(
            f"Y range: "
            f"{bounding_box['y_min']} - "
            f"{bounding_box['y_max']}"
        )

        print(
            f"Z range: "
            f"{bounding_box['z_min']} - "
            f"{bounding_box['z_max']}"
        )

        print(
            f"Tumor volume: "
            f"{tumor_volume_mm3:.2f} mm^3"
        )

        print(
            f"Tumor volume: "
            f"{tumor_volume_cm3:.2f} cm^3"
        )

    else:

        print(
            "No tumor candidate detected."
        )

    # --------------------------------------------------------
    # Save segmentation
    # --------------------------------------------------------

    segmentation_output_path = os.path.join(
        segmented_dir,
        f"{patient_id}_segmented.nii.gz"
    )

    save_segmentation(
        final_mask,
        reference_image,
        segmentation_output_path
    )

    print()
    print(
        f"Segmentation saved:\n"
        f"{segmentation_output_path}"
    )

    # --------------------------------------------------------
    # Save localization
    # --------------------------------------------------------

    localization_output_path = os.path.join(
        segmented_dir,
        f"{patient_id}_localization.json"
    )

    localization_data = {
        "patient_id": patient_id,
        "run_id": config["run_id"],
        "tumor_detected": bounding_box["found"],
        "bounding_box_voxel": bounding_box,
        "tumor_volume_mm3": tumor_volume_mm3,
        "tumor_volume_cm3": tumor_volume_cm3
    }

    with open(
        localization_output_path,
        "w"
    ) as file:

        json.dump(
            localization_data,
            file,
            indent=4
        )

    print(
        f"Localization data saved:\n"
        f"{localization_output_path}"
    )

    # --------------------------------------------------------
    # Diagnostic
    # --------------------------------------------------------

    diagnostic_output_path = os.path.join(
        image_dir,
        f"{patient_id}_segmentation.png"
    )

    best_slice = save_segmentation_diagnostic(
        flair,
        t2,
        t1ce,
        final_mask,
        diagnostic_output_path,
        patient_id
    )

    print(
        f"Diagnostic image saved:\n"
        f"{diagnostic_output_path}"
    )

    print()
    print("=" * 65)
    print("SEGMENTATION COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(
        f"Representative slice: {best_slice}"
    )

    # Return the actual generated NIfTI path
    # to the next module.
    return segmentation_output_path


# ============================================================
# OPTIONAL STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    print(
        "Run segmentation through main.py."
    )