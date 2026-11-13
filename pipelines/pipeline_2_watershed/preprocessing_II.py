import os
import sys
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PIPELINE_NAME = "Pipeline_2_Watershed"
METHOD_TAG = "watershed"

SEQUENCES = [
    "t1",
    "t1ce",
    "t2",
    "flair"
]


# ============================================================
# INTENSITY NORMALIZATION
# ============================================================

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Performs robust volume-level percentile intensity normalization.
    Only non-zero voxels (brain tissue) are considered when calculating
    the 1st and 99th percentiles to avoid skew from the air background.

    Output range: 0.0 - 1.0 (float32)
    """
    volume = volume.astype(np.float32)
    brain_mask = volume > 0

    if not np.any(brain_mask):
        return np.zeros_like(volume, dtype=np.float32)

    brain_voxels = volume[brain_mask]
    p1, p99 = np.percentile(brain_voxels, (1.0, 99.0))

    if p99 <= p1:
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[brain_mask] = 1.0
        return normalized

    clipped = np.clip(volume, p1, p99)
    normalized = (clipped - p1) / (p99 - p1)
    normalized[~brain_mask] = 0.0

    return normalized.astype(np.float32)


# ============================================================
# EDGE-PRESERVING DENOISING (BILATERAL + MEDIAN)
# ============================================================

def denoise_volume(
    volume: np.ndarray,
    kernel_size: int = 3,
    use_bilateral: bool = True
) -> np.ndarray:
    """
    Edge-preserving filtering tailored for Watershed gradient relief.
    Combines median blur (for impulse noise removal) with bilateral filtering
    (which smooths tissue interiors while preserving sharp anatomical boundaries).
    """
    num_slices, height, width = volume.shape
    denoised = np.zeros_like(volume, dtype=np.float32)

    for z in range(num_slices):
        slice_2d = volume[z]
        if np.max(slice_2d) <= 0:
            denoised[z] = 0.0
            continue

        slice_uint8 = np.clip(slice_2d * 255.0, 0, 255).astype(np.uint8)

        # Step 1: Median blur for impulse noise
        med_filtered = cv2.medianBlur(slice_uint8, kernel_size)

        # Step 2: Bilateral filter for edge-preserving smoothing
        if use_bilateral:
            filtered_uint8 = cv2.bilateralFilter(
                med_filtered,
                d=5,
                sigmaColor=25,
                sigmaSpace=5
            )
        else:
            filtered_uint8 = med_filtered

        filtered = filtered_uint8.astype(np.float32) / 255.0
        filtered[slice_2d <= 0] = 0.0
        denoised[z] = filtered

    return denoised


# ============================================================
# CONTRAST-LIMITED ADAPTIVE HISTOGRAM EQUALIZATION (CLAHE)
# ============================================================

def apply_clahe_volume(
    volume: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE slice-by-slice to enhance local tissue contrast
    between hyperintense tumor margins and healthy brain parenchyma.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    num_slices, height, width = volume.shape
    enhanced_volume = np.zeros_like(volume, dtype=np.float32)

    for z in range(num_slices):
        slice_2d = volume[z]
        if np.max(slice_2d) <= 0:
            enhanced_volume[z] = 0.0
            continue

        slice_uint8 = np.clip(slice_2d * 255.0, 0, 255).astype(np.uint8)
        enhanced_uint8 = clahe.apply(slice_uint8)

        enhanced = enhanced_uint8.astype(np.float32) / 255.0
        enhanced[slice_2d <= 0] = 0.0
        enhanced_volume[z] = enhanced

    return enhanced_volume


# ============================================================
# NIFTI I/O
# ============================================================

def save_nifti(
    volume: np.ndarray,
    reference_image: sitk.Image,
    output_path: str
):
    """
    Saves a 3D numpy volume as a NIfTI image, copying spatial metadata
    (origin, spacing, direction matrix) from the reference image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_image = sitk.GetImageFromArray(volume)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, output_path)


# ============================================================
# DIAGNOSTIC VISUALIZATION
# ============================================================

def save_preprocessing_diagnostic(
    normalized_volume: np.ndarray,
    denoised_volume: np.ndarray,
    clahe_volume: np.ndarray,
    sequence_name: str,
    output_path: str,
    patient_id: str
):
    """
    Generates a 3-panel diagnostic visualization of the slice with
    the largest brain area comparing Normalized, Denoised (Bilateral),
    and CLAHE-enhanced views.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    slice_areas = [
        np.sum(normalized_volume[z] > 0)
        for z in range(normalized_volume.shape[0])
    ]
    slice_idx = int(np.argmax(slice_areas))

    norm_slice = normalized_volume[slice_idx]
    denoised_slice = denoised_volume[slice_idx]
    clahe_slice = clahe_volume[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(norm_slice, cmap="gray")
    axes[0].set_title(f"{sequence_name.upper()} - Normalized")
    axes[0].axis("off")

    axes[1].imshow(denoised_slice, cmap="gray")
    axes[1].set_title(f"Bilateral Denoised (Edge-Preserving)")
    axes[1].axis("off")

    axes[2].imshow(clahe_slice, cmap="gray")
    axes[2].set_title(f"CLAHE Enhanced")
    axes[2].axis("off")

    plt.suptitle(
        f"Pipeline 2.0 (Watershed) Preprocessing: {patient_id} - "
        f"{sequence_name.upper()} - Slice {slice_idx}",
        fontsize=13,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# PROCESS ONE SEQUENCE
# ============================================================

def preprocess_sequence(
    sequence_name: str,
    config: dict
):
    patient_id = config["patient_id"]
    data_dir = config["data_dir"]
    denoised_dir = config["denoised_dir"]
    clahe_dir = config["clahe_dir"]
    image_dir = config["image_dir"]

    input_path = os.path.join(
        data_dir,
        f"{patient_id}_{sequence_name}.nii"
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Standard and method-tagged output paths
    denoised_output_path = os.path.join(
        denoised_dir,
        f"{patient_id}_{sequence_name}_denoised_{METHOD_TAG}.nii.gz"
    )
    clahe_output_path = os.path.join(
        clahe_dir,
        f"{patient_id}_{sequence_name}_clahe_{METHOD_TAG}.nii.gz"
    )
    diagnostic_output_path = os.path.join(
        image_dir,
        f"{patient_id}_{sequence_name}_preprocessing_{METHOD_TAG}.png"
    )

    print()
    print("-" * 65)
    print(f"PREPROCESSING (WATERSHED PIPELINE) - {sequence_name.upper()}")
    print("-" * 65)

    sitk_image = sitk.ReadImage(input_path)
    volume = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

    print(f"Volume shape      : {volume.shape}")
    normalized = normalize_volume(volume)
    print(f"Normalized range  : {normalized.min():.4f} - {normalized.max():.4f}")

    denoised = denoise_volume(normalized, kernel_size=3, use_bilateral=True)
    clahe = apply_clahe_volume(denoised)

    # Save primary tagged outputs
    save_nifti(denoised, sitk_image, denoised_output_path)
    save_nifti(clahe, sitk_image, clahe_output_path)

    # Also save standard names for full cross-compatibility
    standard_denoised = os.path.join(denoised_dir, f"{patient_id}_{sequence_name}_denoised.nii.gz")
    standard_clahe = os.path.join(clahe_dir, f"{patient_id}_{sequence_name}_clahe.nii.gz")
    save_nifti(denoised, sitk_image, standard_denoised)
    save_nifti(clahe, sitk_image, standard_clahe)

    save_preprocessing_diagnostic(
        normalized,
        denoised,
        clahe,
        sequence_name,
        diagnostic_output_path,
        patient_id
    )

    print(f"Denoised output   : {denoised_output_path}")
    print(f"CLAHE output      : {clahe_output_path}")
    print(f"Diagnostic image  : {diagnostic_output_path}")


# ============================================================
# PIPELINE FUNCTION
# ============================================================

def run_preprocessing(config: dict):
    print()
    print("=" * 65)
    print("PHASE 1: BRAIN MRI PREPROCESSING (PIPELINE 2.0 - WATERSHED)")
    print("=" * 65)
    print(f"Patient ID        : {config['patient_id']}")
    print(f"Run ID            : {config['run_id']}")
    print(f"Data directory    : {config['data_dir']}")

    for sequence in SEQUENCES:
        preprocess_sequence(sequence, config)

    print()
    print("=" * 65)
    print("PHASE 1 COMPLETE: ALL 4 MRI SEQUENCES PROCESSED")
    print("=" * 65)


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(_THIS_DIR)

    print()
    print("=" * 75)
    print("   PIPELINE 2.0 (WATERSHED) - STANDALONE PREPROCESSING")
    print("=" * 75)

    default_patient = "BraTS20_Training_001"
    user_input = input(f"Enter patient ID (default: {default_patient}): ").strip()
    patient_id = user_input if user_input else default_patient

    config = {
        "project_dir": PROJECT_DIR,
        "patient_id": patient_id,
        "run_id": f"standalone_watershed_prep_{patient_id}",
        "data_dir": os.path.join(PROJECT_DIR, "data", "Training", patient_id),
        "denoised_dir": os.path.join(PROJECT_DIR, "Outputs", "Pipeline_2_Watershed", "Denoised", patient_id),
        "clahe_dir": os.path.join(PROJECT_DIR, "Outputs", "Pipeline_2_Watershed", "CLAHE", patient_id),
        "image_dir": os.path.join(PROJECT_DIR, "Results", "Pipeline_2_Watershed", "Images", patient_id),
    }

    if not os.path.isdir(config["data_dir"]):
        print(f"Error: Data directory not found: {config['data_dir']}")
        sys.exit(1)

    run_preprocessing(config)
    print(f"\nStandalone preprocessing complete for {patient_id}!")
