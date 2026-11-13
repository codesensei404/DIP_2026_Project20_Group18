import os
import sys
import json
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage.segmentation as seg


# ============================================================
# CONFIGURATION
# ============================================================

PIPELINE_NAME = "Pipeline_2_Watershed"
METHOD_TAG = "watershed"

# ── Tunable parameters (Pipeline 2 — Watershed) ─────────────
# Results analysis (n=100):
#   Problem : Worst HD95 (28mm), high variance, bimodal outcomes.
#             35% over-segmentation from noisy gradient ridges
#             and an unlabeled zone that's too large.
#   Fix     : Raise gradient sigma to smooth out micro-ridges,
#             tighten the external marker threshold (was 75th pct →
#             raise to 70th so healthy tissue is marked earlier),
#             reduce satellite acceptance distance.
#
# Recommended search space (see tune.py):
#   GRADIENT_SIGMA         : [1.0, 1.5, 2.0]        (was 1.0)
#   FLAIR_PERCENTILE       : [83, 85, 87]            (was 85)
#   T2_PERCENTILE          : [83, 85, 87]            (was 85)
#   T1CE_PERCENTILE        : [88, 90, 92]            (was 90)
#   HEALTHY_PERCENTILE     : [65, 70, 75]            (was 75)
#   HEALTHY_DIST_VOXELS    : [4, 5, 6]               (was 4)
#   MIN_COMPONENT_VOLUME   : [150, 200, 300]         (was 200)
#   SATELLITE_DIST_MM      : [8, 10, 12]             (was 12)
# ─────────────────────────────────────────────────────────────
P2_GRADIENT_SIGMA       = 1.68   # tuned: Optuna best 1.6785 (probe Dice 0.6880)
P2_FLAIR_PERCENTILE     = 85     # unchanged
P2_T2_PERCENTILE        = 85     # unchanged
P2_T1CE_PERCENTILE      = 90     # unchanged
P2_HEALTHY_PERCENTILE   = 74     # tuned: from 70 → marks healthy brain earlier
P2_HEALTHY_DIST_VOXELS  = 3      # tuned: tightened from 5 → larger healthy zone
MIN_COMPONENT_VOLUME_MM3 = 200.0
P2_SATELLITE_DIST_MM    = 10.46  # tuned: Optuna best 10.4626


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired volume not found:\n{path}\n"
            f"Please ensure preprocessing has been executed first."
        )

    image = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    return volume, image


# ============================================================
# VERIFY SEQUENCES
# ============================================================

def verify_sequences(images: dict):
    names = list(images.keys())
    reference_name = names[0]
    reference_image = images[reference_name][1]
    reference_shape = images[reference_name][0].shape
    reference_spacing = reference_image.GetSpacing()
    reference_origin = reference_image.GetOrigin()
    reference_direction = reference_image.GetDirection()

    for name in names[1:]:
        volume = images[name][0]
        image = images[name][1]

        if volume.shape != reference_shape:
            raise ValueError(
                f"Shape mismatch: {reference_name}: {reference_shape} vs {name}: {volume.shape}"
            )

        if not np.allclose(image.GetSpacing(), reference_spacing):
            raise ValueError(f"Spacing mismatch between {reference_name} and {name}.")

        if not np.allclose(image.GetOrigin(), reference_origin):
            raise ValueError(f"Origin mismatch between {reference_name} and {name}.")

        if not np.allclose(image.GetDirection(), reference_direction):
            raise ValueError(f"Direction mismatch between {reference_name} and {name}.")

    print("All multi-sequence MRI volumes verified: matching shapes and spatial geometry.")


# ============================================================
# 3D BRAIN MASK CREATION & BASAL SLICE FILTERING
# ============================================================

def create_brain_mask(
    t1: np.ndarray,
    t1ce: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray
) -> np.ndarray:
    """
    Constructs an anatomically tight 3D skull-stripped brain mask
    from the multi-sequence union, followed by 3D closing and hole-filling.
    Also suppresses basal skull base and orbital slices with low brain area.
    """
    raw_brain_mask = (t1 > 0) | (t1ce > 0) | (t2 > 0) | (flair > 0)
    structure = ndi.generate_binary_structure(3, 2)

    brain_mask = ndi.binary_closing(raw_brain_mask, structure=structure, iterations=1)
    brain_mask = ndi.binary_fill_holes(brain_mask)

    # Dynamic basal slice filtering:
    # Slices below z=12 or containing < 15% max brain area contain orbital fat,
    # neck, sinuses, and skull base bony ridges that cause false-positive flooding.
    slice_areas = np.sum(brain_mask, axis=(1, 2))
    max_area = np.max(slice_areas)
    valid_slices = np.where(slice_areas >= 0.15 * max_area)[0]
    basal_cutoff = max(12, int(valid_slices[0])) if len(valid_slices) > 0 else 12

    refined_brain = brain_mask.copy()
    refined_brain[:basal_cutoff, :, :] = False

    return refined_brain.astype(bool)


# ============================================================
# MULTI-MODAL GRADIENT (TOPOGRAPHIC ELEVATION MAP)
# ============================================================

def compute_multimodal_gradient(
    t1: np.ndarray,
    t1ce: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray,
    brain_mask: np.ndarray,
    sigma: float = P2_GRADIENT_SIGMA
) -> np.ndarray:
    """
    Synthesizes a 3D topographic elevation landscape for watershed flooding.
    In BraTS, Whole Tumor (edema + core) boundaries are sharpest on FLAIR and T2,
    while enhancing rim boundaries are highlighted by T1ce.
    """
    # Multi-sequence contrast fusion
    fused_intensity = (
        0.50 * flair +
        0.30 * t2 +
        0.20 * t1ce
    )

    # 3D Gaussian gradient magnitude to produce smooth topographic ridges
    gradient_magnitude = ndi.gaussian_gradient_magnitude(
        fused_intensity,
        sigma=sigma
    )

    # Restrict gradient strictly to intracranial volume
    gradient_magnitude[~brain_mask] = 0.0

    return gradient_magnitude.astype(np.float32)


# ============================================================
# WATERSHED MARKER GENERATION
# ============================================================

def generate_watershed_markers(
    flair: np.ndarray,
    t2: np.ndarray,
    t1ce: np.ndarray,
    brain_mask: np.ndarray
) -> tuple:
    """
    Generates internal (tumor core) and external (healthy brain + air background)
    markers for Marker-Controlled Watershed Segmentation.

    Label Conventions:
      0: Unlabeled / Transition zone (to be flooded by watershed)
      1: External Marker (Healthy brain tissue and air background)
      2: Internal Marker (Definite tumor core and hyperintense seed)
    """
    brain_voxels_flair = flair[brain_mask]
    brain_voxels_t2 = t2[brain_mask]
    brain_voxels_t1ce = t1ce[brain_mask]

    flair_th = float(np.percentile(brain_voxels_flair, P2_FLAIR_PERCENTILE))
    t2_th = float(np.percentile(brain_voxels_t2, P2_T2_PERCENTILE))
    t1ce_th = float(np.percentile(brain_voxels_t1ce, P2_T1CE_PERCENTILE))

    # Multi-modal tumor candidate region
    flair_t2_cand = (flair >= flair_th) & (t2 >= t2_th)
    t1ce_flair_cand = (t1ce >= t1ce_th) & (flair >= flair_th)
    tumor_candidate = (flair_t2_cand | t1ce_flair_cand) & brain_mask

    # Morphological opening to break thin bridge connections to non-tumor structures
    structure = ndi.generate_binary_structure(3, 2)
    cand_opened = ndi.binary_opening(tumor_candidate, structure=structure, iterations=1)
    cand_opened = ndi.binary_closing(cand_opened, structure=structure, iterations=1)

    # Internal Markers: Erode opened candidate to capture high-confidence tumor seeds
    internal_seeds = ndi.binary_erosion(
        cand_opened,
        structure=structure,
        iterations=1
    )

    # Fallback: if erosion wiped out very small tumors, retain un-eroded candidate
    if np.sum(internal_seeds) < 50:
        internal_seeds = cand_opened if np.sum(cand_opened) > 0 else tumor_candidate

    # External Markers:
    # Healthy brain: normal intensity FLAIR (< P2_HEALTHY_PERCENTILE)
    # located safely away from any tumor candidate (distance > P2_HEALTHY_DIST_VOXELS)
    dist_from_cand = ndi.distance_transform_edt(~cand_opened)
    healthy_threshold = float(np.percentile(brain_voxels_flair, P2_HEALTHY_PERCENTILE))
    healthy_brain = brain_mask & (flair < healthy_threshold) & (dist_from_cand > P2_HEALTHY_DIST_VOXELS)

    # Construct the marker volume
    markers = np.zeros(flair.shape, dtype=np.int32)
    markers[~brain_mask] = 1       # Non-brain = background (1)
    markers[healthy_brain] = 1     # Healthy parenchyma = background (1)
    markers[internal_seeds] = 2    # High-confidence tumor seeds = foreground (2)

    return markers, internal_seeds, healthy_brain, tumor_candidate


# ============================================================
# MARKER-CONTROLLED WATERSHED SEGMENTATION
# ============================================================

def run_marker_controlled_watershed(
    elevation_map: np.ndarray,
    markers: np.ndarray,
    brain_mask: np.ndarray
) -> np.ndarray:
    """
    Executes 3D marker-controlled watershed flooding from designated
    internal and external seeds over the multi-modal gradient elevation surface.
    """
    watershed_labels = seg.watershed(
        elevation_map,
        markers=markers,
        mask=brain_mask
    )

    # Extract the segmented tumor label (Marker 2)
    tumor_mask = (watershed_labels == 2)
    return tumor_mask


# ============================================================
# 3D MORPHOLOGICAL REFINEMENT
# ============================================================

def refine_candidate_mask(mask: np.ndarray) -> np.ndarray:
    """
    Applies 3D morphological closing to smooth boundary irregularities
    and fills internal cavities (e.g., necrotic tissue within the tumor).
    """
    structure = ndi.generate_binary_structure(3, 2)
    refined = ndi.binary_closing(mask, structure=structure, iterations=1)
    refined = ndi.binary_fill_holes(refined)
    return refined


# ============================================================
# 3D ANATOMICAL CONNECTIVITY FILTERING
# ============================================================

def filter_connected_components(
    mask: np.ndarray,
    reference_image: sitk.Image,
    min_volume_mm3: float = MIN_COMPONENT_VOLUME_MM3
) -> np.ndarray:
    """
    Filters 3D connected components using an anatomical clustering prior:
    Primary glioma mass is retained; secondary satellites are retained only if
    they are substantial (>= 20% size of primary) and proximate (<= 12 mm distance).
    All isolated false-positive fragments (ventricular CSF, orbits, choroid plexus) are rejected.
    """
    structure = ndi.generate_binary_structure(3, 2)
    labels, num_components = ndi.label(mask, structure=structure)

    if num_components == 0:
        return mask

    spacing = reference_image.GetSpacing()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    sizes = [(c, np.sum(labels == c)) for c in range(1, num_components + 1)]
    sizes.sort(key=lambda x: x[1], reverse=True)
    top1_lbl, top1_sz = sizes[0]

    # Retain the primary dominant mass
    clean_mask = (labels == top1_lbl)
    dist_from_top = ndi.distance_transform_edt(~clean_mask)
    kept = 1

    # Check secondary satellites
    for c, sz in sizes[1:]:
        phys_vol = sz * voxel_volume_mm3
        if phys_vol >= min_volume_mm3 and sz >= 0.20 * top1_sz:
            min_dist = float(np.min(dist_from_top[labels == c]))
            if min_dist <= P2_SATELLITE_DIST_MM:
                clean_mask |= (labels == c)
                kept += 1

    print(f"3D Connected Components: {num_components} found, {kept} retained (primary mass + proximal satellites).")
    return clean_mask


# ============================================================
# BOUNDING BOX & VOLUME
# ============================================================

def calculate_bounding_box(tumor_mask: np.ndarray) -> dict:
    coords = np.where(tumor_mask)
    if coords[0].size == 0:
        return {"found": False}

    return {
        "found": True,
        "z_min": int(coords[0].min()),
        "z_max": int(coords[0].max()),
        "y_min": int(coords[1].min()),
        "y_max": int(coords[1].max()),
        "x_min": int(coords[2].min()),
        "x_max": int(coords[2].max()),
    }


def calculate_tumor_volume(
    tumor_mask: np.ndarray,
    reference_image: sitk.Image
) -> tuple:
    voxel_count = int(np.sum(tumor_mask))
    spacing = reference_image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    volume_mm3 = float(voxel_count * voxel_volume)
    volume_cm3 = float(volume_mm3 / 1000.0)
    return volume_mm3, volume_cm3


# ============================================================
# SAVE SEGMENTATION & DIAGNOSTIC VISUALIZATION
# ============================================================

def save_segmentation(
    tumor_mask: np.ndarray,
    reference_image: sitk.Image,
    output_path: str
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_uint8 = tumor_mask.astype(np.uint8)
    image = sitk.GetImageFromArray(mask_uint8)
    image.CopyInformation(reference_image)
    sitk.WriteImage(image, output_path)


def save_segmentation_diagnostic(
    flair: np.ndarray,
    gradient: np.ndarray,
    markers: np.ndarray,
    tumor_mask: np.ndarray,
    output_path: str,
    patient_id: str
) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    slice_areas = np.sum(tumor_mask, axis=(1, 2))
    best_slice = int(np.argmax(slice_areas)) if np.max(slice_areas) > 0 else int(tumor_mask.shape[0] // 2)

    flair_slice = flair[best_slice]
    grad_slice = gradient[best_slice]
    marker_slice = markers[best_slice]
    mask_slice = tumor_mask[best_slice]

    # Create RGB contour overlay on FLAIR
    flair_uint8 = np.clip(flair_slice * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(flair_uint8, cv2.COLOR_GRAY2RGB)
    mask_uint8 = (mask_slice.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    axes[0].imshow(flair_slice, cmap="gray")
    axes[0].set_title("FLAIR Input")
    axes[0].axis("off")

    axes[1].imshow(grad_slice, cmap="magma")
    axes[1].set_title("3D Gradient (Elevation Map)")
    axes[1].axis("off")

    # RGB Marker view: Internal seeds = green, Healthy/Background = blue
    marker_rgb = np.zeros((*marker_slice.shape, 3), dtype=np.uint8)
    marker_rgb[marker_slice == 1] = [50, 50, 200]   # Blue = Background
    marker_rgb[marker_slice == 2] = [50, 220, 50]   # Green = Tumor seeds
    axes[2].imshow(marker_rgb)
    axes[2].set_title("Watershed Markers (Seeds)")
    axes[2].axis("off")

    axes[3].imshow(mask_slice, cmap="gray")
    axes[3].set_title("Watershed Tumor Mask")
    axes[3].axis("off")

    axes[4].imshow(overlay)
    axes[4].set_title("Tumor Boundary Overlay")
    axes[4].axis("off")

    plt.suptitle(
        f"Pipeline 2.0 (Marker-Controlled Watershed): {patient_id} - Representative Slice {best_slice}",
        fontsize=13,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return best_slice


# ============================================================
# PIPELINE FUNCTION
# ============================================================

def run_segmentation(config: dict) -> str:
    patient_id = config["patient_id"]
    denoised_dir = config["denoised_dir"]
    segmented_dir = config["segmented_dir"]
    image_dir = config["image_dir"]

    print()
    print("=" * 65)
    print("PHASE 2: 3D MARKER-CONTROLLED WATERSHED SEGMENTATION")
    print("=" * 65)
    print(f"Patient ID        : {patient_id}")
    print(f"Run ID            : {config['run_id']}")

    # Load denoised volumes (supports both tagged and standard names)
    def resolve_path(seq):
        tagged = os.path.join(denoised_dir, f"{patient_id}_{seq}_denoised_{METHOD_TAG}.nii.gz")
        standard = os.path.join(denoised_dir, f"{patient_id}_{seq}_denoised.nii.gz")
        return tagged if os.path.exists(tagged) else standard

    print("\nLoading multi-sequence volumes...")
    t1, t1_img = load_nifti(resolve_path("t1"))
    t1ce, t1ce_img = load_nifti(resolve_path("t1ce"))
    t2, t2_img = load_nifti(resolve_path("t2"))
    flair, flair_img = load_nifti(resolve_path("flair"))

    images = {
        "T1": (t1, t1_img),
        "T1ce": (t1ce, t1ce_img),
        "T2": (t2, t2_img),
        "FLAIR": (flair, flair_img)
    }
    verify_sequences(images)
    reference_image = flair_img

    # 1. 3D Brain mask with basal artifact filtering
    print("\nConstructing intracranial 3D brain mask (with basal artifact filtering)...")
    brain_mask = create_brain_mask(t1, t1ce, t2, flair)
    print(f"Intracranial brain voxels: {np.sum(brain_mask)}")

    # 2. Multi-modal gradient surface
    print("\nComputing multi-modal topographic elevation landscape...")
    gradient = compute_multimodal_gradient(t1, t1ce, t2, flair, brain_mask, sigma=1.0)

    # 3. Watershed markers
    print("\nGenerating internal tumor seeds and external background markers...")
    markers, tumor_seeds, healthy_brain, cand = generate_watershed_markers(flair, t2, t1ce, brain_mask)
    print(f"Tumor seed voxels (Internal Marker 2): {np.sum(markers == 2)}")
    print(f"Healthy/Background voxels (External Marker 1): {np.sum(markers == 1)}")

    # 4. Marker-controlled watershed flooding
    print("\nExecuting 3D marker-controlled watershed transformation...")
    raw_tumor_mask = run_marker_controlled_watershed(gradient, markers, brain_mask)
    print(f"Raw watershed tumor voxels: {np.sum(raw_tumor_mask)}")

    # 5. 3D Morphological refinement & Anatomical Connected Components
    print("\nApplying 3D morphological refinement & anatomical component filtering...")
    refined_mask = refine_candidate_mask(raw_tumor_mask)
    final_mask = filter_connected_components(refined_mask, reference_image, MIN_COMPONENT_VOLUME_MM3)
    final_voxels = int(np.sum(final_mask))
    print(f"Final segmented tumor voxels: {final_voxels}")

    # 6. Localization & Volume
    bbox = calculate_bounding_box(final_mask)
    vol_mm3, vol_cm3 = calculate_tumor_volume(final_mask, reference_image)

    print("\nTumor Localization:")
    if bbox["found"]:
        print(f"  Voxel Bounding Box: X=[{bbox['x_min']}-{bbox['x_max']}], Y=[{bbox['y_min']}-{bbox['y_max']}], Z=[{bbox['z_min']}-{bbox['z_max']}]")
        print(f"  Segmented Volume  : {vol_mm3:.2f} mm³ ({vol_cm3:.2f} cm³)")
    else:
        print("  No tumor mass detected.")

    # 7. Save outputs (both tagged and standard names)
    tagged_seg_path = os.path.join(segmented_dir, f"{patient_id}_segmentation_{METHOD_TAG}.nii.gz")
    standard_seg_path = os.path.join(segmented_dir, f"{patient_id}_segmented.nii.gz")
    save_segmentation(final_mask, reference_image, tagged_seg_path)
    save_segmentation(final_mask, reference_image, standard_seg_path)

    loc_data = {
        "patient_id": patient_id,
        "run_id": config["run_id"],
        "pipeline": PIPELINE_NAME,
        "method": METHOD_TAG,
        "tumor_detected": bbox["found"],
        "bounding_box_voxel": bbox,
        "tumor_volume_mm3": vol_mm3,
        "tumor_volume_cm3": vol_cm3,
        "tumor_voxels": final_voxels
    }

    tagged_loc_path = os.path.join(segmented_dir, f"{patient_id}_localization_{METHOD_TAG}.json")
    standard_loc_path = os.path.join(segmented_dir, f"{patient_id}_localization.json")
    for loc_file in [tagged_loc_path, standard_loc_path]:
        with open(loc_file, "w") as f:
            json.dump(loc_data, f, indent=4)

    diagnostic_path = os.path.join(image_dir, f"{patient_id}_segmentation_{METHOD_TAG}.png")
    best_slice = save_segmentation_diagnostic(flair, gradient, markers, final_mask, diagnostic_path, patient_id)

    print(f"\nSegmentation saved : {tagged_seg_path}")
    print(f"Localization saved : {tagged_loc_path}")
    print(f"Diagnostic saved   : {diagnostic_path}")
    print(f"Representative slice: {best_slice}")

    print()
    print("=" * 65)
    print("PHASE 2 COMPLETE: WATERSHED SEGMENTATION FINISHED")
    print("=" * 65)

    return tagged_seg_path


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(_THIS_DIR)

    print()
    print("=" * 75)
    print("   PIPELINE 2.0 (WATERSHED) - STANDALONE SEGMENTATION")
    print("=" * 75)

    default_patient = "BraTS20_Training_001"
    user_input = input(f"Enter patient ID (default: {default_patient}): ").strip()
    patient_id = user_input if user_input else default_patient

    config = {
        "project_dir": PROJECT_DIR,
        "patient_id": patient_id,
        "run_id": f"standalone_watershed_seg_{patient_id}",
        "data_dir": os.path.join(PROJECT_DIR, "data", "Training", patient_id),
        "denoised_dir": os.path.join(PROJECT_DIR, "Outputs", "Pipeline_2_Watershed", "Denoised", patient_id),
        "segmented_dir": os.path.join(PROJECT_DIR, "Outputs", "Pipeline_2_Watershed", "Segmented", patient_id),
        "image_dir": os.path.join(PROJECT_DIR, "Results", "Pipeline_2_Watershed", "Images", patient_id),
    }

    run_segmentation(config)
    print(f"\nStandalone watershed segmentation complete for {patient_id}!")
