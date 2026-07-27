import os

import PIL.Image
from PIL.Image import Resampling
import nibabel as nib
from typing import Optional
from typing import List, Dict, Any
import numpy as np

import scipy.ndimage


def read_image(path: str) -> PIL.Image.Image:
    """
    Read an image from a local path into a PIL RGB image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local image file not found: {path}")
    image = PIL.Image.open(path)

    return image.convert("RGB")


def read_nifti(path: str):
    """
    Read a NIfTI file (.nii or .nii.gz) from a local path.
    Returns a nibabel.Nifti1Image object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local NIfTI file not found: {path}")
    return nib.load(path)


def image_resize(
    image: PIL.Image.Image, 
    min_size: Optional[int] = None, 
    max_size: Optional[int] = None
) -> PIL.Image.Image:
    """
    Resizes a PIL image to fit within given min and max size constraints.
    Maintains aspect ratio unless extreme ratios force clamping.
    """
    if not isinstance(image, PIL.Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

    if min_size is None and max_size is None:
        return image

    if min_size is not None and max_size is not None and min_size > max_size:
        raise ValueError(f"min_size ({min_size}) cannot be greater than max_size ({max_size}).")

    w, h = image.size

    # Check compliance
    compliant = True
    if min_size is not None and (w < min_size or h < min_size):
        compliant = False
    if max_size is not None and (w > max_size or h > max_size):
        compliant = False
    if compliant:
        return image

    # Proportional scaling
    target_w, target_h = w, h

    if max_size is not None and max(target_w, target_h) > max_size:
        scale = max_size / max(target_w, target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    if min_size is not None and min(target_w, target_h) < min_size:
        scale = min_size / min(target_w, target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    # Resolve conflicts (extreme aspect ratio)
    conflict = max_size is not None and max(target_w, target_h) > max_size
    if conflict:
        final_w = min(max(target_w, min_size or 0), max_size or target_w)
        final_h = min(max(target_h, min_size or 0), max_size or target_h)
        print(
            f"Warning: Aspect ratio conflict for {w}x{h} with min={min_size}, max={max_size}. "
            f"Forcing resize to {final_w}x{final_h}."
        )
        target_w, target_h = final_w, final_h

    if (target_w, target_h) != (w, h):
        print(f"Info: Resizing image from {w}x{h} to {target_w}x{target_h}")
        return image.resize((target_w, target_h), resample=Resampling.LANCZOS)
    else:
        return image
    

def nifti_resize(
    nifti_image: nib.Nifti1Image,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    interpolation_order: int = 3,
) -> nib.Nifti1Image:
    """
    Resizes the in-plane spatial dimensions (X, Y) of a 3D NIfTI image.

    This function preserves the number of slices (Z-axis). The in-plane dimensions
    are resized to fit within min_size and max_size constraints while maintaining
    aspect ratio.

    It correctly resamples the voxel data and updates the affine matrix to reflect
    the new in-plane voxel sizes, preserving the physical dimensions and orientation.

    Args:
        nifti_image (nib.Nifti1Image): The 3D NIfTI image to resize.
        min_size (Optional[int]): The minimum allowed size for the X and Y dimensions.
        max_size (Optional[int]): The maximum allowed size for the X and Y dimensions.
        interpolation_order (int): The order of spline interpolation.
            - 0: Nearest-neighbor (good for masks/labels)
            - 1: Linear
            - 3: Cubic (good for general medical images)

    Returns:
        nib.Nifti1Image: A new, resized NIfTI image object.
    """
    data = nifti_image.get_fdata()
    original_affine = nifti_image.affine
    original_shape = data.shape
    
    # --- 1. Validate Input is 3D ---
    if data.ndim != 3:
        raise ValueError(f"Unsupported NIfTI dimension: {data.ndim}. This function only supports 3D images.")

    w, h, z = original_shape
    
    # --- 2. Calculate Target In-Plane Dimensions ---
    target_w, target_h = w, h
    if max_size is not None and max(target_w, target_h) > max_size:
        scale = max_size / max(target_w, target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    if min_size is not None and min(target_w, target_h) < min_size:
        scale = min_size / min(target_w, target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)
    
    # If no resizing is needed, return the original image
    if (target_w, target_h) == (w, h):
        return nifti_image

    print(f"Info: Resizing NIfTI in-plane from {(w, h)} to {(target_w, target_h)}. Slices remain at {z}.")

    # --- 3. Calculate Zoom Factors and Resample Data ---
    # The zoom factor for the slice dimension (Z) is 1.0, meaning no change.
    zoom_factors = [
        target_w / w,
        target_h / h,
        1.0, 
    ]

    # Perform the resampling
    resampled_data = scipy.ndimage.zoom(data, zoom_factors, order=interpolation_order)

    # --- 4. Update the Affine Matrix ---
    # The scaling factor for the Z-axis is 1.0, preserving its voxel size.
    scale_matrix = np.array([
        [w / target_w, 0, 0, 0],
        [0, h / target_h, 0, 0],
        [0, 0, 1.0, 0],  # No change in Z-axis scale
        [0, 0, 0, 1]
    ])
    
    new_affine = original_affine @ scale_matrix

    # --- 5. Create and Return New NIfTI Object ---
    new_nifti = nib.Nifti1Image(resampled_data, new_affine, header=nifti_image.header)

    return new_nifti


def nifti_to_image_slices(nifti_image: nib.Nifti1Image, num_slices: int, window_level: float = 40, window_width: float = 400) -> List[PIL.Image.Image]:
    """
    Transforms a NIfTI image into a list of RGB PIL Image objects (axial view).

    The function extracts slices along the axial (Z) axis. If the NIfTI image
    has more slices than num_slices, it extracts them uniformly. If it has
    fewer, it uses all available slices.

    Args:
        nifti_image (nibabel.Nifti1Image): The NIfTI image object loaded by nibabel.
        num_slices (int): The target number of slices to extract.

    Returns:
        List[Image.Image]: A list of slices as RGB PIL.Image.Image objects.
    """
    # Get the image data as a numpy array
    # NIfTI data is usually in (x, y, z) format, where z is the axial slice
    data = nifti_image.get_fdata()

    if data.ndim != 3:
        raise ValueError(f"Unsupported NIfTI dimension: {data.ndim}. Expected 3D.")

    total_slices = data.shape[2]
    
    if total_slices == 0:
        return []

    # Determine which slice indices to extract
    if total_slices <= num_slices:
        # Use all available slices
        slice_indices = np.arange(total_slices)
    else:
        # Uniformly sample slice indices
        slice_indices = np.linspace(0, total_slices - 1, num_slices).round().astype(int)

    print(f"Extracting {num_slices} slices from {total_slices} total slices.")

    # Process and convert each selected slice
    image_slices = []
    for slice_idx in slice_indices:
        # Extract the 2D slice
        slice_data = data[:, :, slice_idx]
        
        # --- Image Orientation ---
        # NIfTI slices might need rotation to match standard medical viewing conventions.
        # A 90-degree rotation is common for axial slices.
        slice_data = np.rot90(slice_data)

        lower_bound = window_level - (window_width / 2)
        upper_bound = window_level + (window_width / 2)
        
        # Clip the data to the window range
        slice_windowed = np.clip(slice_data, lower_bound, upper_bound)
        
        # Normalize the windowed data to 0-1 range
        # This is the correct way to normalize after clipping
        if upper_bound - lower_bound > 0:
            slice_normalized = (slice_windowed - lower_bound) / (upper_bound - lower_bound)
        else:
            slice_normalized = np.zeros_like(slice_windowed)

        # Scale to 0-255 and convert to 8-bit unsigned integer
        slice_uint8 = (slice_normalized * 255).astype(np.uint8)

        # --- Convert to RGB PIL Image ---
        # Create a grayscale ('L' mode) image first
        pil_image_gray =PIL.Image.fromarray(slice_uint8, mode='L')
        
        # Convert the grayscale image to RGB
        # This duplicates the grayscale channel into R, G, and B channels
        pil_image_rgb = pil_image_gray.convert('RGB')
        
        image_slices.append(pil_image_rgb)
        
    return image_slices