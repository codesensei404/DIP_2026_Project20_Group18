import os
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_PATIENT_ID = "BraTS20_Training_001"

SEQUENCES = [
    "t1",
    "t1ce",
    "t2",
    "flair"
]


# ============================================================
# VOLUME NORMALIZATION
# ============================================================

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Performs volume-level intensity normalization.

    Only non-zero voxels are considered when calculating the
    1st and 99th percentiles.

    Output range:
        0.0 - 1.0
    """

    volume = volume.astype(
        np.float32
    )

    brain_mask = volume > 0

    if not np.any(brain_mask):

        return np.zeros_like(
            volume,
            dtype=np.float32
        )

    brain_voxels = volume[
        brain_mask
    ]

    p1, p99 = np.percentile(
        brain_voxels,
        (1, 99)
    )

    if p99 <= p1:

        normalized = np.zeros_like(
            volume,
            dtype=np.float32
        )

        normalized[
            brain_mask
        ] = 1.0

        return normalized

    clipped = np.clip(
        volume,
        p1,
        p99
    )

    normalized = (
        clipped - p1
    ) / (
        p99 - p1
    )

    normalized[
        ~brain_mask
    ] = 0.0

    return normalized.astype(
        np.float32
    )


# ============================================================
# MEDIAN DENOISING
# ============================================================

def denoise_volume(
    volume: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:

    num_slices, height, width = volume.shape

    denoised = np.zeros_like(
        volume,
        dtype=np.float32
    )

    for z in range(num_slices):

        slice_2d = volume[z]

        slice_uint8 = np.clip(
            slice_2d * 255.0,
            0,
            255
        ).astype(np.uint8)

        filtered_uint8 = cv2.medianBlur(
            slice_uint8,
            kernel_size
        )

        filtered = (
            filtered_uint8.astype(
                np.float32
            ) / 255.0
        )

        filtered[
            slice_2d <= 0
        ] = 0.0

        denoised[z] = filtered

    return denoised


# ============================================================
# CLAHE
# ============================================================

def apply_clahe_volume(
    volume: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    num_slices, height, width = volume.shape

    clahe_volume = np.zeros_like(
        volume,
        dtype=np.float32
    )

    for z in range(num_slices):

        slice_2d = volume[z]

        slice_uint8 = np.clip(
            slice_2d * 255.0,
            0,
            255
        ).astype(np.uint8)

        enhanced_uint8 = clahe.apply(
            slice_uint8
        )

        enhanced = (
            enhanced_uint8.astype(
                np.float32
            ) / 255.0
        )

        enhanced[
            slice_2d <= 0
        ] = 0.0

        clahe_volume[z] = enhanced

    return clahe_volume


# ============================================================
# SAVE NIFTI
# ============================================================

def save_nifti(
    volume: np.ndarray,
    reference_image: sitk.Image,
    output_path: str
):

    output_image = sitk.GetImageFromArray(
        volume
    )

    output_image.CopyInformation(
        reference_image
    )

    sitk.WriteImage(
        output_image,
        output_path
    )


# ============================================================
# DIAGNOSTIC VISUALIZATION
# ============================================================

def save_preprocessing_diagnostic(
    normalized_volume,
    denoised_volume,
    clahe_volume,
    sequence_name,
    output_path,
    patient_id
):

    slice_areas = [
        np.sum(
            normalized_volume[z] > 0
        )
        for z in range(
            normalized_volume.shape[0]
        )
    ]

    slice_idx = int(
        np.argmax(slice_areas)
    )

    normalized_slice = normalized_volume[
        slice_idx
    ]

    denoised_slice = denoised_volume[
        slice_idx
    ]

    clahe_slice = clahe_volume[
        slice_idx
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        normalized_slice,
        cmap="gray"
    )

    axes[0].set_title(
        f"{sequence_name.upper()} - Normalized"
    )

    axes[0].axis("off")

    axes[1].imshow(
        denoised_slice,
        cmap="gray"
    )

    axes[1].set_title(
        "Median Denoised"
    )

    axes[1].axis("off")

    axes[2].imshow(
        clahe_slice,
        cmap="gray"
    )

    axes[2].set_title(
        "CLAHE"
    )

    axes[2].axis("off")

    plt.suptitle(
        f"{patient_id} - "
        f"Preprocessing - "
        f"{sequence_name.upper()} - "
        f"Slice {slice_idx}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PROCESS ONE SEQUENCE
# ============================================================

def preprocess_sequence(
    sequence_name,
    config
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

    denoised_output_path = os.path.join(
        denoised_dir,
        f"{patient_id}_{sequence_name}_denoised.nii.gz"
    )

    clahe_output_path = os.path.join(
        clahe_dir,
        f"{patient_id}_{sequence_name}_clahe.nii.gz"
    )

    diagnostic_output_path = os.path.join(
        image_dir,
        f"{patient_id}_{sequence_name}_preprocessing.png"
    )

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"{input_path}"
        )

    print()
    print("=" * 65)
    print(
        f"PROCESSING {sequence_name.upper()}"
    )
    print("=" * 65)

    sitk_image = sitk.ReadImage(
        input_path
    )

    volume = sitk.GetArrayFromImage(
        sitk_image
    ).astype(np.float32)

    print(
        f"Volume shape: {volume.shape}"
    )

    normalized = normalize_volume(
        volume
    )

    print(
        f"Normalized range: "
        f"{normalized.min():.4f} - "
        f"{normalized.max():.4f}"
    )

    denoised = denoise_volume(
        normalized,
        kernel_size=3
    )

    clahe = apply_clahe_volume(
        denoised
    )

    save_nifti(
        denoised,
        sitk_image,
        denoised_output_path
    )

    save_nifti(
        clahe,
        sitk_image,
        clahe_output_path
    )

    save_preprocessing_diagnostic(
        normalized,
        denoised,
        clahe,
        sequence_name,
        diagnostic_output_path,
        patient_id
    )

    print(
        f"Denoised output:\n"
        f"{denoised_output_path}"
    )

    print(
        f"CLAHE output:\n"
        f"{clahe_output_path}"
    )

    print(
        f"Diagnostic image:\n"
        f"{diagnostic_output_path}"
    )


# ============================================================
# PIPELINE FUNCTION
# ============================================================

def run_preprocessing(config):

    print()
    print("=" * 65)
    print("BRAIN MRI PREPROCESSING")
    print("=" * 65)

    print(
        f"Patient: {config['patient_id']}"
    )

    print(
        f"Run: {config['run_id']}"
    )

    print(
        f"Data directory:\n"
        f"{config['data_dir']}"
    )

    for sequence in SEQUENCES:

        preprocess_sequence(
            sequence,
            config
        )

    print()
    print("=" * 65)
    print("ALL FOUR MRI SEQUENCES PROCESSED SUCCESSFULLY")
    print("=" * 65)


# ============================================================
# OPTIONAL STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    print(
        "This module is normally executed "
        "through main.py."
    )

    print(
        "Run: python main.py"
    )