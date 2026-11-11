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

PIPELINE_NAME = "Pipeline_3_Hybrid"
METHOD_TAG = "hybrid"
MIN_COMPONENT_VOLUME_MM3 = 200.0


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
    raw_brain = (t1 > 0) | (t1ce > 0) | (t2 > 0) | (flair > 0)
    structure = ndi.generate_binary_structure(3, 2)
    brain_mask = ndi.binary_closing(raw_brain, structure=structure, iterations=1)
    brain_mask = ndi.binary_fill_holes(brain_mask)

    # Dynamic basal slice filtering:
    # Eliminates skull base, orbits, and neck slices where tumor never resides in BraTS
    slice_areas = np.sum(brain_mask, axis=(1, 2))
    max_area = np.max(slice_areas)
    valid_slices = np.where(slice_areas >= 0.15 * max_area)[0]
    basal_cutoff = max(12, int(valid_slices[0])) if len(valid_slices) > 0 else 12

    refined_brain = brain_mask.copy()
    refined_brain[:basal_cutoff, :, :] = False
    return refined_brain.astype(bool)


# ============================================================
# HYBRID SEGMENTATION (THRESHOLD-GUIDED WATERSHED)
# ============================================================

def run_hybrid_segmentation(
    t1: np.ndarray,
    t1ce: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray,
    brain_mask: np.ndarray
) -> tuple:
    """
    Hybrid Segmentation Architecture (Fine-Tuned):
    1. Multi-Sequence Percentile Thresholding generates high-precision core seeds.
    2. Primary mass extraction eliminates secondary false-positive seeds before dilation.
    3. Geodesic distance-intensity envelope constrains search band to true peritumoral edema.
    4. Marker-Controlled Watershed floods along 3D gradient ridges within the search band.
    5. Morphological Consensus Fusion combines the core and boundary delineations.
    """
    brain_voxels_flair = flair[brain_mask]
    brain_voxels_t2 = t2[brain_mask]
    brain_voxels_t1ce = t1ce[brain_mask]

    # --- Phase A: Thresholding Core Extraction ---
    flair_th_high = float(np.percentile(brain_voxels_flair, 85))
    t2_th_high = float(np.percentile(brain_voxels_t2, 85))
    t1ce_th_high = float(np.percentile(brain_voxels_t1ce, 90))

    flair_t2_core = (flair >= flair_th_high) & (t2 >= t2_th_high)
    t1ce_flair_core = (t1ce >= t1ce_th_high) & (flair >= flair_th_high)
    threshold_core = (flair_t2_core | t1ce_flair_core) & brain_mask

    structure = ndi.generate_binary_structure(3, 2)
    threshold_core = ndi.binary_opening(threshold_core, structure=structure, iterations=1)
    threshold_core = ndi.binary_closing(threshold_core, structure=structure, iterations=2)
    threshold_core = ndi.binary_fill_holes(threshold_core)

    # Filter disjoint components on the core before expanding into edema band
    lbl, num = ndi.label(threshold_core, structure=structure)
    if num > 0:
        sizes = [(c, np.sum(lbl == c)) for c in range(1, num + 1)]
        sizes.sort(key=lambda x: x[1], reverse=True)
        top1_lbl, top1_sz = sizes[0]
        refined_core = (lbl == top1_lbl)
        dist_from_top1 = ndi.distance_transform_edt(~refined_core)
        for c, sz in sizes[1:]:
            if sz >= 0.20 * top1_sz and np.min(dist_from_top1[lbl == c]) <= 12.0:
                refined_core |= (lbl == c)
    else:
        refined_core = threshold_core

    # --- Phase B: Constrained Edema Search Band ---
    # Prevents watershed from leaking into far ventricles or contralateral hemisphere
    dist_from_core = ndi.distance_transform_edt(~refined_core)
    flair_th_edema = float(np.percentile(brain_voxels_flair, 80))
    edema_band = (dist_from_core <= 4.0) & (flair >= flair_th_edema) & brain_mask
    allowed_region = refined_core | edema_band

    # --- Phase C: Elevation Landscape & Watershed Flooding ---
    fused_intensity = 0.50 * flair + 0.30 * t2 + 0.20 * t1ce
    gradient = ndi.gaussian_gradient_magnitude(fused_intensity, sigma=1.0)
    gradient[~brain_mask] = 0.0

    # Internal markers: eroded threshold core
    internal_seeds = ndi.binary_erosion(refined_core, structure=structure, iterations=1)
    if np.sum(internal_seeds) < 50:
        internal_seeds = refined_core

    # External markers: background outside allowed region & healthy tissue
    markers = np.zeros(flair.shape, dtype=np.int32)
    markers[~allowed_region] = 1
    healthy_tissue = allowed_region & (dist_from_core > 3.0) & (flair < float(np.percentile(brain_voxels_flair, 75)))
    markers[healthy_tissue] = 1
    markers[internal_seeds] = 2

    # Watershed within the constrained boundary
    watershed_labels = seg.watershed(gradient, markers=markers, mask=allowed_region)
    watershed_tumor = (watershed_labels == 2)

    # --- Phase D: Consensus Fusion & Cleanup ---
    hybrid_tumor = refined_core | watershed_tumor
    hybrid_tumor = ndi.binary_closing(hybrid_tumor, structure=structure, iterations=1)
    hybrid_tumor = ndi.binary_fill_holes(hybrid_tumor)

    return hybrid_tumor, refined_core, watershed_tumor, gradient, allowed_region


# ============================================================
# 3D ANATOMICAL CONNECTIVITY FILTERING
# ============================================================

def filter_connected_components(
    mask: np.ndarray,
    reference_image: sitk.Image,
    min_volume_mm3: float = MIN_COMPONENT_VOLUME_MM3
) -> np.ndarray:
    """
    Applies anatomical clustering prior: retains primary tumor mass and proximate satellites.
    Rejects isolated non-tumor hyperintensities (ventricles, choroid plexus, orbital fat).
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

    clean_mask = (labels == top1_lbl)
    dist_from_top = ndi.distance_transform_edt(~clean_mask)
    kept = 1

    for c, sz in sizes[1:]:
        phys_vol = sz * voxel_volume_mm3
        if phys_vol >= min_volume_mm3 and sz >= 0.20 * top1_sz:
            if float(np.min(dist_from_top[labels == c])) <= 10.0:
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
    threshold_core: np.ndarray,
    gradient: np.ndarray,
    final_mask: np.ndarray,
    output_path: str,
    patient_id: str
) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    slice_areas = np.sum(final_mask, axis=(1, 2))
    best_slice = int(np.argmax(slice_areas)) if np.max(slice_areas) > 0 else int(final_mask.shape[0] // 2)

    flair_slice = flair[best_slice]
    core_slice = threshold_core[best_slice]
    grad_slice = gradient[best_slice]
    final_slice = final_mask[best_slice]

    flair_uint8 = np.clip(flair_slice * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(flair_uint8, cv2.COLOR_GRAY2RGB)
    mask_uint8 = (final_slice.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    axes[0].imshow(flair_slice, cmap="gray")
    axes[0].set_title("FLAIR Input")
    axes[0].axis("off")

    axes[1].imshow(core_slice, cmap="Blues")
    axes[1].set_title("Phase A: Threshold Core")
    axes[1].axis("off")

    axes[2].imshow(grad_slice, cmap="magma")
    axes[2].set_title("Phase B/C: Gradient Surface")
    axes[2].axis("off")

    axes[3].imshow(final_slice, cmap="gray")
    axes[3].set_title("Phase D: Consensus Mask")
    axes[3].axis("off")

    axes[4].imshow(overlay)
    axes[4].set_title("Hybrid Contour Overlay")
    axes[4].axis("off")

    plt.suptitle(
        f"Pipeline 3.0 (Hybrid): {patient_id} - Representative Slice {best_slice}",
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
    print("PHASE 2: HYBRID (THRESHOLD-GUIDED WATERSHED) SEGMENTATION")
    print("=" * 65)
    print(f"Patient ID        : {patient_id}")
    print(f"Run ID            : {config['run_id']}")

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

    # 2. Hybrid Segmentation (Threshold core -> Bounded Watershed flooding)
    print("\nExecuting Hybrid segmentation (Threshold Core -> Constrained Watershed Flooding)...")
    raw_hybrid, threshold_core, ws_mask, grad, allowed_region = run_hybrid_segmentation(
        t1, t1ce, t2, flair, brain_mask
    )
    print(f"Threshold core voxels       : {np.sum(threshold_core)}")
    print(f"Constrained watershed voxels: {np.sum(ws_mask)}")
    print(f"Combined hybrid voxels      : {np.sum(raw_hybrid)}")

    # 3. Anatomical Connected Component Filtering
    print("\nApplying anatomical connected component filtering (primary mass + proximal satellites)...")
    final_mask = filter_connected_components(raw_hybrid, reference_image, MIN_COMPONENT_VOLUME_MM3)
    final_voxels = int(np.sum(final_mask))
    print(f"Final segmented tumor voxels: {final_voxels}")

    # 4. Localization & Volume
    bbox = calculate_bounding_box(final_mask)
    vol_mm3, vol_cm3 = calculate_tumor_volume(final_mask, reference_image)

    print("\nTumor Localization:")
    if bbox["found"]:
        print(f"  Voxel Bounding Box: X=[{bbox['x_min']}-{bbox['x_max']}], Y=[{bbox['y_min']}-{bbox['y_max']}], Z=[{bbox['z_min']}-{bbox['z_max']}]")
        print(f"  Segmented Volume  : {vol_mm3:.2f} mm³ ({vol_cm3:.2f} cm³)")
    else:
        print("  No tumor mass detected.")

    # 5. Save outputs (both tagged and standard names)
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
    best_slice = save_segmentation_diagnostic(flair, threshold_core, grad, final_mask, diagnostic_path, patient_id)

    print(f"\nSegmentation saved : {tagged_seg_path}")
    print(f"Localization saved : {tagged_loc_path}")
    print(f"Diagnostic saved   : {diagnostic_path}")
    print(f"Representative slice: {best_slice}")

    print()
    print("=" * 65)
    print("PHASE 2 COMPLETE: HYBRID SEGMENTATION FINISHED")
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
    print("   PIPELINE 3.0 (HYBRID) - STANDALONE SEGMENTATION")
    print("=" * 75)

    default_patient = "BraTS20_Training_001"
    user_input = input(f"Enter patient ID (default: {default_patient}): ").strip()
    patient_id = user_input if user_input else default_patient

    config = {
        "project_dir": PROJECT_DIR,
        "patient_id": patient_id,
        "run_id": f"standalone_hybrid_seg_{patient_id}",
        "data_dir": os.path.join(PROJECT_DIR, "data", "Training", patient_id),
        "denoised_dir": os.path.join(PROJECT_DIR, "Outputs", "Pipeline_3_Hybrid", "Denoised", patient_id),
        "segmented_dir": os.path.join(PROJECT_DIR, "Outputs", "Pipeline_3_Hybrid", "Segmented", patient_id),
        "image_dir": os.path.join(PROJECT_DIR, "Results", "Pipeline_3_Hybrid", "Images", patient_id),
    }

    run_segmentation(config)
    print(f"\nStandalone hybrid segmentation complete for {patient_id}!")
