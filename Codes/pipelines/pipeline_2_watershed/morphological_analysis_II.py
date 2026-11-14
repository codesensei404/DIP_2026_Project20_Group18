import os
import sys
import json
import csv
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area


# ============================================================
# CONFIGURATION
# ============================================================

PIPELINE_NAME = "Pipeline_2_Watershed"
METHOD_TAG = "watershed"


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path: str):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Segmentation mask not found:\n{mask_path}")

    image = sitk.ReadImage(mask_path)
    array = sitk.GetArrayFromImage(image)
    mask = array > 0

    spacing_xyz = image.GetSpacing()
    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])

    return mask, image, spacing_xyz, spacing_zyx


# ============================================================
# BASIC GEOMETRY
# ============================================================

def calculate_basic_geometry(
    mask: np.ndarray,
    spacing_xyz: tuple
) -> dict:
    tumor_voxels = int(np.sum(mask))
    voxel_volume_mm3 = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    tumor_volume_mm3 = float(tumor_voxels * voxel_volume_mm3)
    tumor_volume_cm3 = float(tumor_volume_mm3 / 1000.0)

    coordinates = np.argwhere(mask)

    if len(coordinates) == 0:
        return {
            "tumor_voxel_count": 0,
            "voxel_volume_mm3": voxel_volume_mm3,
            "tumor_volume_mm3": 0.0,
            "tumor_volume_cm3": 0.0,
            "bbox_x_voxels": 0,
            "bbox_y_voxels": 0,
            "bbox_z_voxels": 0,
            "bbox_x_mm": 0.0,
            "bbox_y_mm": 0.0,
            "bbox_z_mm": 0.0,
            "bbox_volume_mm3": 0.0,
            "centroid_x_mm": 0.0,
            "centroid_y_mm": 0.0,
            "centroid_z_mm": 0.0,
            "min_x_voxel": 0, "max_x_voxel": 0,
            "min_y_voxel": 0, "max_y_voxel": 0,
            "min_z_voxel": 0, "max_z_voxel": 0,
        }

    z_min, y_min, x_min = coordinates.min(axis=0)
    z_max, y_max, x_max = coordinates.max(axis=0)

    bbox_size_voxels = np.array([
        z_max - z_min + 1,
        y_max - y_min + 1,
        x_max - x_min + 1
    ])

    bbox_size_mm = np.array([
        bbox_size_voxels[2] * spacing_xyz[0],
        bbox_size_voxels[1] * spacing_xyz[1],
        bbox_size_voxels[0] * spacing_xyz[2]
    ])

    bbox_volume_mm3 = float(np.prod(bbox_size_mm))
    centroid_zyx = coordinates.mean(axis=0)
    centroid_xyz_mm = np.array([
        centroid_zyx[2] * spacing_xyz[0],
        centroid_zyx[1] * spacing_xyz[1],
        centroid_zyx[0] * spacing_xyz[2]
    ])

    return {
        "tumor_voxel_count": tumor_voxels,
        "voxel_volume_mm3": voxel_volume_mm3,
        "tumor_volume_mm3": tumor_volume_mm3,
        "tumor_volume_cm3": tumor_volume_cm3,
        "bbox_x_voxels": int(bbox_size_voxels[2]),
        "bbox_y_voxels": int(bbox_size_voxels[1]),
        "bbox_z_voxels": int(bbox_size_voxels[0]),
        "bbox_x_mm": float(bbox_size_mm[0]),
        "bbox_y_mm": float(bbox_size_mm[1]),
        "bbox_z_mm": float(bbox_size_mm[2]),
        "bbox_volume_mm3": bbox_volume_mm3,
        "centroid_x_mm": float(centroid_xyz_mm[0]),
        "centroid_y_mm": float(centroid_xyz_mm[1]),
        "centroid_z_mm": float(centroid_xyz_mm[2]),
        "min_x_voxel": int(x_min), "max_x_voxel": int(x_max),
        "min_y_voxel": int(y_min), "max_y_voxel": int(y_max),
        "min_z_voxel": int(z_min), "max_z_voxel": int(z_max)
    }


# ============================================================
# SLICE-WISE AREA ANALYSIS
# ============================================================

def calculate_slice_area(
    mask: np.ndarray,
    spacing_xyz: tuple
) -> dict:
    pixel_area_mm2 = float(spacing_xyz[0] * spacing_xyz[1])
    areas_pixels = np.sum(mask, axis=(1, 2))
    tumor_slices = np.where(areas_pixels > 0)[0]

    if len(tumor_slices) == 0:
        return {
            "tumor_slice_count": 0,
            "max_area_pixels": 0,
            "max_area_mm2": 0.0,
            "mean_area_pixels": 0.0,
            "mean_area_mm2": 0.0,
            "total_projected_area_pixels": 0,
            "total_projected_area_mm2": 0.0,
            "max_area_slice": None,
            "pixel_area_mm2": pixel_area_mm2
        }

    tumor_areas = areas_pixels[tumor_slices]
    max_index = np.argmax(tumor_areas)
    max_area_pixels = int(tumor_areas[max_index])
    max_area_mm2 = float(max_area_pixels * pixel_area_mm2)
    mean_area_pixels = float(np.mean(tumor_areas))
    mean_area_mm2 = float(mean_area_pixels * pixel_area_mm2)
    total_proj_pixels = int(np.sum(tumor_areas))
    total_proj_mm2 = float(total_proj_pixels * pixel_area_mm2)

    return {
        "pixel_area_mm2": pixel_area_mm2,
        "tumor_slice_count": int(len(tumor_slices)),
        "max_area_pixels": max_area_pixels,
        "max_area_mm2": max_area_mm2,
        "mean_area_pixels": mean_area_pixels,
        "mean_area_mm2": mean_area_mm2,
        "total_projected_area_pixels": total_proj_pixels,
        "total_projected_area_mm2": total_proj_mm2,
        "max_area_slice": int(tumor_slices[max_index])
    }


# ============================================================
# PHYSICAL EXTENT
# ============================================================

def calculate_physical_extent(
    mask: np.ndarray,
    spacing_xyz: tuple
) -> dict:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return {"extent_x_mm": 0.0, "extent_y_mm": 0.0, "extent_z_mm": 0.0}

    z_range = coordinates[:, 0].max() - coordinates[:, 0].min() + 1
    y_range = coordinates[:, 1].max() - coordinates[:, 1].min() + 1
    x_range = coordinates[:, 2].max() - coordinates[:, 2].min() + 1

    return {
        "extent_x_mm": float(x_range * spacing_xyz[0]),
        "extent_y_mm": float(y_range * spacing_xyz[1]),
        "extent_z_mm": float(z_range * spacing_xyz[2])
    }


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def calculate_components(
    mask: np.ndarray,
    spacing_xyz: tuple
) -> dict:
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num_components = ndimage.label(mask, structure=structure)

    if num_components == 0:
        return {
            "connected_components": 0,
            "largest_component_volume_mm3": 0.0,
            "largest_component_fraction": 0.0
        }

    component_sizes = np.bincount(labeled.ravel())[1:]
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    component_volumes = component_sizes * voxel_volume_mm3
    largest_volume = float(np.max(component_volumes))
    total_volume = float(np.sum(component_volumes))

    return {
        "connected_components": int(num_components),
        "largest_component_volume_mm3": largest_volume,
        "largest_component_fraction": float(largest_volume / total_volume) if total_volume > 0 else 0.0
    }


# ============================================================
# SURFACE AREA (MARCHING CUBES)
# ============================================================

def calculate_surface_area(
    mask: np.ndarray,
    spacing_xyz: tuple
) -> dict:
    if np.sum(mask) == 0:
        return {"surface_area_mm2": 0.0}

    try:
        vertices, faces, _, _ = marching_cubes(
            mask.astype(np.float32),
            level=0.5,
            spacing=(spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        )
        surface_area = mesh_surface_area(vertices, faces)
        return {"surface_area_mm2": float(surface_area)}
    except Exception:
        return {"surface_area_mm2": None}


# ============================================================
# SHAPE DESCRIPTORS
# ============================================================

def calculate_shape_descriptors(
    mask: np.ndarray,
    geometry: dict,
    surface: dict
) -> dict:
    volume = geometry["tumor_volume_mm3"]
    surface_area = surface["surface_area_mm2"]

    if volume <= 0:
        return {
            "equivalent_spherical_diameter_mm": 0.0,
            "sphericity": 0.0,
            "compactness": 0.0,
            "bounding_box_fill_ratio": 0.0
        }

    equivalent_diameter = float((6.0 * volume / np.pi) ** (1.0 / 3.0))

    if surface_area is not None and surface_area > 0:
        sphericity = float((np.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / surface_area)
        compactness = float((36.0 * np.pi * (volume ** 2)) / (surface_area ** 3))
    else:
        sphericity = 0.0
        compactness = 0.0

    bbox_volume = geometry["bbox_volume_mm3"]
    fill_ratio = float(volume / bbox_volume) if bbox_volume > 0 else 0.0

    return {
        "equivalent_spherical_diameter_mm": equivalent_diameter,
        "sphericity": sphericity,
        "compactness": compactness,
        "bounding_box_fill_ratio": fill_ratio
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: dict,
    mask_path: str,
    config: dict
) -> tuple:
    patient_id = config["patient_id"]
    run_id = config["run_id"]
    table_dir = config["table_dir"]

    os.makedirs(table_dir, exist_ok=True)

    tagged_csv = os.path.join(table_dir, f"{patient_id}_morphological_analysis_{METHOD_TAG}.csv")
    tagged_json = os.path.join(table_dir, f"{patient_id}_morphological_analysis_{METHOD_TAG}.json")
    standard_csv = os.path.join(table_dir, f"{patient_id}_morphological_analysis.csv")
    standard_json = os.path.join(table_dir, f"{patient_id}_morphological_analysis.json")

    # CSV write
    for out_csv in [tagged_csv, standard_csv]:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value"])
            for key, val in results.items():
                writer.writerow([key, val])

    # JSON write
    results_meta = {
        "patient_id": patient_id,
        "run_id": run_id,
        "pipeline": PIPELINE_NAME,
        "method": METHOD_TAG,
        "mask_file": mask_path,
        **results
    }

    for out_json in [tagged_json, standard_json]:
        with open(out_json, "w") as f:
            json.dump(results_meta, f, indent=4)

    print()
    print("-" * 65)
    print(f"MORPHOLOGICAL ANALYSIS ({METHOD_TAG.upper()}) SUMMARY")
    print("-" * 65)
    for key, val in results.items():
        print(f"  {key:40s}: {val}")
    print()
    print(f"Saved: {tagged_csv}")
    print(f"Saved: {tagged_json}")

    return tagged_csv, tagged_json


# ============================================================
# PIPELINE FUNCTION
# ============================================================

def analyze_mask(
    mask_path: str,
    config: dict
) -> dict:
    print()
    print("=" * 65)
    print("PHASE 3: MORPHOLOGICAL ANALYSIS (PIPELINE 2.0 - WATERSHED)")
    print("=" * 65)
    print(f"Loading mask: {mask_path}")

    mask, image, spacing_xyz, spacing_zyx = load_mask(mask_path)
    print(f"Mask shape [Z, Y, X] : {mask.shape}")
    print(f"Voxel spacing [X, Y, Z] (mm): {spacing_xyz}")

    geometry = calculate_basic_geometry(mask, spacing_xyz)
    slice_analysis = calculate_slice_area(mask, spacing_xyz)
    physical_extent = calculate_physical_extent(mask, spacing_xyz)
    components = calculate_components(mask, spacing_xyz)
    surface = calculate_surface_area(mask, spacing_xyz)
    shape = calculate_shape_descriptors(mask, geometry, surface)

    results = {}
    results.update(geometry)
    results.update(slice_analysis)
    results.update(physical_extent)
    results.update(components)
    results.update(surface)
    results.update(shape)

    save_results(results, mask_path, config)

    print()
    print("=" * 65)
    print("PHASE 3 COMPLETE: MORPHOLOGICAL ANALYSIS FINISHED")
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
    print("  PIPELINE 2.0 (WATERSHED) - STANDALONE MORPHOLOGICAL ANALYSIS")
    print("=" * 75)

    default_patient = "BraTS20_Training_001"
    user_input = input(f"Enter patient ID (default: {default_patient}): ").strip()
    patient_id = user_input if user_input else default_patient

    mask_path = os.path.join(
        PROJECT_DIR,
        "Outputs",
        "Pipeline_2_Watershed",
        "Segmented",
        patient_id,
        f"{patient_id}_segmentation_{METHOD_TAG}.nii.gz"
    )

    if not os.path.isfile(mask_path):
        # Fallback to standard name
        mask_path = os.path.join(
            PROJECT_DIR,
            "Outputs",
            "Pipeline_2_Watershed",
            "Segmented",
            patient_id,
            f"{patient_id}_segmented.nii.gz"
        )

    config = {
        "project_dir": PROJECT_DIR,
        "patient_id": patient_id,
        "run_id": f"standalone_morph_{METHOD_TAG}_{patient_id}",
        "table_dir": os.path.join(PROJECT_DIR, "Results", "Pipeline_2_Watershed", "Tables", patient_id),
    }

    analyze_mask(mask_path, config)
    print(f"\nStandalone morphological analysis complete for {patient_id}!")
