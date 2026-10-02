import os
import json
import csv

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):

    if not os.path.exists(mask_path):

        raise FileNotFoundError(
            f"Segmentation mask not found:\n"
            f"{mask_path}"
        )

    image = sitk.ReadImage(
        mask_path
    )

    array = sitk.GetArrayFromImage(
        image
    )

    # Any non-zero voxel = tumor
    mask = array > 0

    spacing_xyz = image.GetSpacing()

    spacing_zyx = np.array([
        spacing_xyz[2],
        spacing_xyz[1],
        spacing_xyz[0]
    ])

    return (
        mask,
        image,
        spacing_xyz,
        spacing_zyx
    )


# ============================================================
# BASIC GEOMETRY
# ============================================================

def calculate_basic_geometry(
    mask,
    spacing_xyz
):

    tumor_voxels = int(
        np.sum(mask)
    )

    voxel_volume_mm3 = (
        spacing_xyz[0] *
        spacing_xyz[1] *
        spacing_xyz[2]
    )

    tumor_volume_mm3 = (
        tumor_voxels *
        voxel_volume_mm3
    )

    tumor_volume_cm3 = (
        tumor_volume_mm3 /
        1000.0
    )

    coordinates = np.argwhere(
        mask
    )

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
            "bbox_volume_mm3": 0.0
        }

    z_min, y_min, x_min = (
        coordinates.min(axis=0)
    )

    z_max, y_max, x_max = (
        coordinates.max(axis=0)
    )

    bbox_size_voxels = np.array([
        z_max - z_min + 1,
        y_max - y_min + 1,
        x_max - x_min + 1
    ])

    bbox_size_mm = np.array([
        bbox_size_voxels[2] *
        spacing_xyz[0],

        bbox_size_voxels[1] *
        spacing_xyz[1],

        bbox_size_voxels[0] *
        spacing_xyz[2]
    ])

    bbox_volume_mm3 = np.prod(
        bbox_size_mm
    )

    centroid_zyx = coordinates.mean(
        axis=0
    )

    centroid_xyz_mm = np.array([
        centroid_zyx[2] *
        spacing_xyz[0],

        centroid_zyx[1] *
        spacing_xyz[1],

        centroid_zyx[0] *
        spacing_xyz[2]
    ])

    return {

        "tumor_voxel_count":
            tumor_voxels,

        "voxel_volume_mm3":
            float(voxel_volume_mm3),

        "tumor_volume_mm3":
            float(tumor_volume_mm3),

        "tumor_volume_cm3":
            float(tumor_volume_cm3),

        "bbox_x_voxels":
            int(bbox_size_voxels[2]),

        "bbox_y_voxels":
            int(bbox_size_voxels[1]),

        "bbox_z_voxels":
            int(bbox_size_voxels[0]),

        "bbox_x_mm":
            float(bbox_size_mm[0]),

        "bbox_y_mm":
            float(bbox_size_mm[1]),

        "bbox_z_mm":
            float(bbox_size_mm[2]),

        "bbox_volume_mm3":
            float(bbox_volume_mm3),

        "centroid_x_mm":
            float(centroid_xyz_mm[0]),

        "centroid_y_mm":
            float(centroid_xyz_mm[1]),

        "centroid_z_mm":
            float(centroid_xyz_mm[2]),

        "min_x_voxel":
            int(x_min),

        "max_x_voxel":
            int(x_max),

        "min_y_voxel":
            int(y_min),

        "max_y_voxel":
            int(y_max),

        "min_z_voxel":
            int(z_min),

        "max_z_voxel":
            int(z_max)
    }


# ============================================================
# SLICE-WISE AREA
# ============================================================

def calculate_slice_area(
    mask,
    spacing_xyz
):

    pixel_area_mm2 = (
        spacing_xyz[0] *
        spacing_xyz[1]
    )

    areas_pixels = np.sum(
        mask,
        axis=(1, 2)
    )

    tumor_slices = np.where(
        areas_pixels > 0
    )[0]

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
            "pixel_area_mm2": float(pixel_area_mm2)
        }

    tumor_areas = areas_pixels[
        tumor_slices
    ]

    max_index = np.argmax(
        tumor_areas
    )

    max_area_pixels = int(
        tumor_areas[max_index]
    )

    max_area_mm2 = (
        max_area_pixels *
        pixel_area_mm2
    )

    mean_area_pixels = float(
        np.mean(tumor_areas)
    )

    mean_area_mm2 = (
        mean_area_pixels *
        pixel_area_mm2
    )

    return {

        "pixel_area_mm2":
            float(pixel_area_mm2),

        "tumor_slice_count":
            int(len(tumor_slices)),

        "max_area_pixels":
            max_area_pixels,

        "max_area_mm2":
            float(max_area_mm2),

        "mean_area_pixels":
            mean_area_pixels,

        "mean_area_mm2":
            float(mean_area_mm2),

        "total_projected_area_pixels":
            int(np.sum(tumor_areas)),

        "total_projected_area_mm2":
            float(
                np.sum(tumor_areas) *
                pixel_area_mm2
            ),

        "max_area_slice":
            int(tumor_slices[max_index])
    }


# ============================================================
# PHYSICAL EXTENT
# ============================================================

def calculate_physical_extent(
    mask,
    spacing_xyz
):

    coordinates = np.argwhere(
        mask
    )

    if len(coordinates) == 0:

        return {
            "extent_x_mm": 0.0,
            "extent_y_mm": 0.0,
            "extent_z_mm": 0.0
        }

    z_range = (
        coordinates[:, 0].max() -
        coordinates[:, 0].min() +
        1
    )

    y_range = (
        coordinates[:, 1].max() -
        coordinates[:, 1].min() +
        1
    )

    x_range = (
        coordinates[:, 2].max() -
        coordinates[:, 2].min() +
        1
    )

    return {

        "extent_x_mm":
            float(x_range * spacing_xyz[0]),

        "extent_y_mm":
            float(y_range * spacing_xyz[1]),

        "extent_z_mm":
            float(z_range * spacing_xyz[2])
    }


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def calculate_components(
    mask,
    spacing_xyz
):

    structure = (
        ndimage.generate_binary_structure(
            3,
            2
        )
    )

    labeled, number_of_components = (
        ndimage.label(
            mask,
            structure=structure
        )
    )

    if number_of_components == 0:

        return {
            "connected_components": 0,
            "largest_component_volume_mm3": 0.0,
            "largest_component_fraction": 0.0
        }

    component_sizes = (
        np.bincount(
            labeled.ravel()
        )[1:]
    )

    voxel_volume_mm3 = np.prod(
        spacing_xyz
    )

    component_volumes = (
        component_sizes *
        voxel_volume_mm3
    )

    largest_volume = float(
        np.max(component_volumes)
    )

    total_volume = float(
        np.sum(component_volumes)
    )

    return {

        "connected_components":
            int(number_of_components),

        "largest_component_volume_mm3":
            largest_volume,

        "largest_component_fraction":
            (
                largest_volume /
                total_volume
                if total_volume > 0
                else 0.0
            )
    }


# ============================================================
# SURFACE AREA
# ============================================================

def calculate_surface_area(
    mask,
    spacing_xyz
):

    if np.sum(mask) == 0:

        return {
            "surface_area_mm2": 0.0
        }

    try:

        vertices, faces, _, _ = (
            marching_cubes(
                mask.astype(
                    np.float32
                ),
                level=0.5,
                spacing=(
                    spacing_xyz[2],
                    spacing_xyz[1],
                    spacing_xyz[0]
                )
            )
        )

        surface_area = (
            mesh_surface_area(
                vertices,
                faces
            )
        )

        return {
            "surface_area_mm2":
                float(surface_area)
        }

    except Exception:

        return {
            "surface_area_mm2": None
        }


# ============================================================
# SHAPE DESCRIPTORS
# ============================================================

def calculate_shape_descriptors(
    mask,
    geometry,
    surface
):

    volume = geometry[
        "tumor_volume_mm3"
    ]

    surface_area = surface[
        "surface_area_mm2"
    ]

    if volume <= 0:

        return {
            "equivalent_spherical_diameter_mm": 0.0,
            "sphericity": 0.0,
            "compactness": 0.0,
            "bounding_box_fill_ratio": 0.0
        }

    equivalent_diameter = (
        (
            6.0 *
            volume /
            np.pi
        )
        ** (1.0 / 3.0)
    )

    if (
        surface_area is not None
        and surface_area > 0
    ):

        sphericity = (
            (
                np.pi ** (1.0 / 3.0)
            ) *
            (
                (6.0 * volume)
                ** (2.0 / 3.0)
            ) /
            surface_area
        )

        compactness = (
            (
                36.0 *
                np.pi *
                volume ** 2
            ) /
            (
                surface_area ** 3
            )
        )

    else:

        sphericity = 0.0
        compactness = 0.0

    bbox_volume = geometry[
        "bbox_volume_mm3"
    ]

    fill_ratio = (
        volume / bbox_volume
        if bbox_volume > 0
        else 0.0
    )

    return {

        "equivalent_spherical_diameter_mm":
            float(equivalent_diameter),

        "sphericity":
            float(sphericity),

        "compactness":
            float(compactness),

        "bounding_box_fill_ratio":
            float(fill_ratio)
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    mask_path,
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

    output_csv = os.path.join(
        table_dir,
        f"{patient_id}_morphological_analysis.csv"
    )

    output_json = os.path.join(
        table_dir,
        f"{patient_id}_morphological_analysis.json"
    )

    with open(
        output_csv,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Parameter",
            "Value"
        ])

        for key, value in results.items():

            writer.writerow([
                key,
                value
            ])

    results_with_metadata = {
        "patient_id": patient_id,
        "run_id": run_id,
        "mask_file": mask_path,
        **results
    }

    with open(
        output_json,
        "w"
    ) as file:

        json.dump(
            results_with_metadata,
            file,
            indent=4
        )

    print()
    print("=" * 65)
    print("MORPHOLOGICAL ANALYSIS")
    print("=" * 65)

    for key, value in results.items():

        print(
            f"{key:45s}: {value}"
        )

    print()
    print(
        f"CSV saved to:\n"
        f"{output_csv}"
    )

    print()
    print(
        f"JSON saved to:\n"
        f"{output_json}"
    )

    return (
        output_csv,
        output_json
    )


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_mask(
    mask_path,
    config
):

    print()
    print(
        "Loading segmentation:"
    )

    print(mask_path)

    mask, image, spacing_xyz, spacing_zyx = (
        load_mask(
            mask_path
        )
    )

    print()
    print(
        "Image shape [Z, Y, X]:",
        mask.shape
    )

    print(
        "Voxel spacing [X, Y, Z] mm:",
        spacing_xyz
    )

    geometry = calculate_basic_geometry(
        mask,
        spacing_xyz
    )

    slice_analysis = calculate_slice_area(
        mask,
        spacing_xyz
    )

    physical_extent = calculate_physical_extent(
        mask,
        spacing_xyz
    )

    components = calculate_components(
        mask,
        spacing_xyz
    )

    surface = calculate_surface_area(
        mask,
        spacing_xyz
    )

    shape = calculate_shape_descriptors(
        mask,
        geometry,
        surface
    )

    results = {}

    results.update(
        geometry
    )

    results.update(
        slice_analysis
    )

    results.update(
        physical_extent
    )

    results.update(
        components
    )

    results.update(
        surface
    )

    results.update(
        shape
    )

    save_results(
        results,
        mask_path,
        config
    )

    return results


# ============================================================
# OPTIONAL STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    print(
        "Run morphological analysis "
        "through main.py."
    )