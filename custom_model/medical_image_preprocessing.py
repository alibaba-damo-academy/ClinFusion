"""
Medical Image Preprocessing Module using MONAI for 3D CT Scans

This module provides preprocessing functions for 3D medical images, specifically
CT scans, to prepare them for input to the Merlin vision encoder.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, List, Literal
from PIL import Image


try:
    import monai
    from monai.transforms import (
        Compose,
        LoadImaged,
        SqueezeDimd,
        EnsureChannelFirstd,
        Spacingd,
        Orientationd,
        ScaleIntensityRanged,
        CropForegroundd,
        Resized,
        ToTensord,
        EnsureTyped,
    )
    from monai.data import Dataset, DataLoader
    from monai.utils import set_determinism
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("MONAI is not installed. Please install it to use medical image preprocessing features.")


def get_ct_preprocessing_transforms(
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_size: Tuple[int, int, int] = (64, 512, 512),
    intensity_window: Tuple[int, int] = (-1000, 1000),  # HU window for CT scans
    use_npy_file = False,
) -> Compose:
    """
    Get preprocessing transforms for CT scans.

    Args:
        target_spacing: Target voxel spacing in mm (depth, height, width)
        target_size: Target image size (depth, height, width)
        intensity_window: Intensity window for CT scans in HU (Hounsfield Units)

    Returns:
        MONAI Compose transform pipeline
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for medical image preprocessing.")

    if not use_npy_file:
        transforms = Compose([
            # Load image
            LoadImaged(keys=["image"]),

            # Ensure channel first format
            EnsureChannelFirstd(keys=["image"]),

            # Normalize orientation to RAS
            Orientationd(keys=["image"], axcodes="RAS"),

            # Resample to target spacing
            Spacingd(
                keys=["image"],
                pixdim=target_spacing,
                mode=("bilinear"),
            ),

            # Scale intensity to the tissue/HU window of interest
            ScaleIntensityRanged(
                keys=["image"],
                a_min=intensity_window[0],
                a_max=intensity_window[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            # Resize to target size
            Resized(
                keys=["image"],
                spatial_size=target_size,
                mode="trilinear",
            ),

            # Convert to tensor
            ToTensord(keys=["image"]),
        ])
    else:

        transforms = Compose([
            # 1. Load the 4D array from the .npy file
            LoadImaged(keys=["image"]),
            
            # 2. Squeeze the unnecessary channel dimension
            SqueezeDimd(keys=["image"], dim=0),
            
            # 3. Add the channel dimension back. This is standard practice.
            EnsureChannelFirstd(keys=["image"]),
            
            # 4. (OPTIONAL BUT RECOMMENDED) Normalize the intensity.
            #    Since we know the range is likely [0, 255], we use that.
            #    This scales the data to [0.0, 1.0].
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0, 
                a_max=1.0, # Use the range you discovered in the diagnostic step!
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            
            # 5. Resize to a fixed grid size. This is your main tool for spatial consistency.
            Resized(
                keys=["image"],
                spatial_size=target_size,
                mode="trilinear",
            ),
            
            # 6. Convert to a PyTorch Tensor
            ToTensord(keys=["image"]),
        ])


    return transforms



def preprocess_ct_image(
    image_path: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_size: Tuple[int, int, int] = (64, 512, 512),
    intensity_window: Tuple[int, int] = (-1000, 1000),
) -> torch.Tensor:
    """
    Preprocess a single CT image for input to Merlin vision encoder.

    Args:
        image_path: Path to the CT image file (DICOM, NIfTI, etc.)
        target_spacing: Target voxel spacing in mm (depth, height, width)
        target_size: Target image size (depth, height, width)
        intensity_window: Intensity window for CT scans in HU (Hounsfield Units)

    Returns:
        Preprocessed image tensor of shape (1, channels, depth, height, width)
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for medical image preprocessing.")

    # Get transforms
    if image_path.endswith(".npy"):
        transforms = get_ct_preprocessing_transforms(
            target_spacing=target_spacing,
            target_size=target_size,
            intensity_window=intensity_window,
            use_npy_file=True,
        ) 
    else:
        transforms = get_ct_preprocessing_transforms(
            target_spacing=target_spacing,
            target_size=target_size,
            intensity_window=intensity_window,
            use_npy_file=False,
        )
    
    ### LoadImaged(keys=["image"])
    # Input: {"image": image_path}
    # Output: {'image': monai.data.meta_tensor.MetaTensor (shape: [512, 512, 361])}
    # import ipdb; ipdb.set_trace()

    # Apply transforms
    data_dict = {"image": image_path}
    processed_data = transforms(data_dict)

    # Extract image tensor
    image_tensor = processed_data["image"]

    # Ensure the tensor has the correct shape (batch, channels, depth, height, width)
    if len(image_tensor.shape) == 4:  # (channels, depth, height, width)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor


def create_medical_image_dataset(
    image_paths: list,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_size: Tuple[int, int, int] = (64, 512, 512),
    intensity_window: Tuple[int, int] = (-1000, 1000),
) -> Dataset:
    """
    Create a MONAI dataset for medical images.

    Args:
        image_paths: List of paths to medical image files
        target_spacing: Target voxel spacing in mm (depth, height, width)
        target_size: Target image size (depth, height, width)
        intensity_window: Intensity window for CT scans in HU (Hounsfield Units)

    Returns:
        MONAI Dataset
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for medical image preprocessing.")

    # Create data dictionary
    data_dicts = [{"image": image_path} for image_path in image_paths]

    # Get transforms
    transforms = get_ct_preprocessing_transforms(
        target_spacing=target_spacing,
        target_size=target_size,
        intensity_window=intensity_window,
    )

    # Create dataset
    dataset = Dataset(data=data_dicts, transform=transforms)

    return dataset


class MedicalImagePreprocessor:
    """
    Medical Image Preprocessor class for handling preprocessing of 3D medical images.

    This class provides a high-level interface for preprocessing CT scans and other
    medical images for use with the Merlin vision encoder.
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Tuple[int, int, int] = (64, 512, 512),
        intensity_window: Tuple[int, int] = (-1000, 1000),
    ):
        """
        Initialize the preprocessor.

        Args:
            target_spacing: Target voxel spacing in mm (depth, height, width)
            target_size: Target image size (depth, height, width)
            intensity_window: Intensity window for CT scans in HU (Hounsfield Units)
        """
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for medical image preprocessing.")

        self.target_spacing = target_spacing
        self.target_size = target_size
        self.intensity_window = intensity_window

        # Create transforms
        self.transforms = get_ct_preprocessing_transforms(
            target_spacing=target_spacing,
            target_size=target_size,
            intensity_window=intensity_window,
        )

    def preprocess_single_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single medical image.

        Args:
            image_path: Path to the medical image file

        Returns:
            Preprocessed image tensor of shape (1, channels, depth, height, width)
        """
        return preprocess_ct_image(
            image_path=image_path,
            target_spacing=self.target_spacing,
            target_size=self.target_size,
            intensity_window=self.intensity_window,
        )

    def preprocess_batch(self, image_paths: list, batch_size: int = 1) -> DataLoader:
        """
        Preprocess a batch of medical images.

        Args:
            image_paths: List of paths to medical image files
            batch_size: Batch size for the data loader

        Returns:
            MONAI DataLoader for batch processing
        """
        dataset = create_medical_image_dataset(
            image_paths=image_paths,
            target_spacing=self.target_spacing,
            target_size=self.target_size,
            intensity_window=self.intensity_window,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader


def ct_preprocessing_monai(
        path,
        target_spacing = (1.0, 1.0, 1.0),
        target_size = (64, 512, 512),
        intensity_window = (-1000, 1000)
    ):
    tensor5d = preprocess_ct_image(
        image_path=path,
        target_spacing=target_spacing,
        target_size=target_size,
        intensity_window=intensity_window,
    )  # (1, C, D, H, W)
    t = tensor5d[:, 0:1].squeeze(0)  # -> (1, D, H, W)


    if t.shape[-3:] != target_size:
        t = F.interpolate(
            t.unsqueeze(0), 
            size = target_size, 
            mode = "trilinear", 
            align_corners = False
        ).squeeze(0)

    return t


def sample_slices_from_volume_to_pil(
    volume_tensor: torch.Tensor,
    num_slices: int = 4,
    random: bool = True,
    seed: Optional[int] = None,
    per_slice_minmax: bool = False,
) -> Union[List[Image.Image], List[List[Image.Image]]]:
    """
    Sample slices from a 3D volume and return them as PIL Images.

    Args:
        volume_tensor: 4D or 5D tensor. Either (C, D, H, W) or (B, C, D, H, W).
        num_slices: Number of slices to sample.
        random: If True, randomly sample; otherwise, uniformly sample across depth.
        seed: Optional RNG seed for reproducible random sampling.
        per_slice_minmax: If True, min-max normalize each slice before converting.
                          If False, assume values already in [0, 1] and just clip.

    Returns:
        - If input is (C, D, H, W): List[PIL.Image.Image] of length num_slices.
        - If input is (B, C, D, H, W): List[List[PIL.Image.Image]] with length B,
          each inner list length num_slices.
    """
    if volume_tensor.ndim == 4:
        # (C, D, H, W)
        C, D, H, W = volume_tensor.shape
        v = volume_tensor
        batched = False
    elif volume_tensor.ndim == 5:
        # (B, C, D, H, W)
        B, C, D, H, W = volume_tensor.shape
        v = volume_tensor
        batched = True
    else:
        raise ValueError(f"Expected 4D (C,D,H,W) or 5D (B,C,D,H,W) tensor, got shape {volume_tensor.shape}")

    if seed is not None and random:
        g = torch.Generator(device=volume_tensor.device).manual_seed(seed)
    else:
        g = None

    if D <= num_slices:
        indices = torch.arange(D, device=volume_tensor.device)
        if D < num_slices:
            pad = indices[-1].repeat(num_slices - D)
            indices = torch.cat([indices, pad])
    else:
        if random:
            perm = torch.randperm(D, generator=g, device=volume_tensor.device)
            indices = perm[:num_slices].sort().values
        else:
            # Uniformly spaced indices across depth
            indices = torch.linspace(0, D - 1, steps=num_slices, device=volume_tensor.device).round().long()
            # Ensure within bounds
            indices = indices.clamp(0, D - 1)

    def tensor_slice_to_pil(slice_tensor: torch.Tensor) -> Image.Image:
        # slice_tensor: (H, W) for 1-channel, or (C, H, W) for multi-channel (C=3)
        if slice_tensor.ndim == 3:
            if slice_tensor.shape[0] == 1:
                slice_tensor = slice_tensor[0]  # (H, W)
                mode = "L"
                arr = slice_tensor
            elif slice_tensor.shape[0] == 3:
                mode = "RGB"
                arr = slice_tensor
            else:
                # For non 1/3 channels, collapse to grayscale by mean
                slice_tensor = slice_tensor.mean(dim=0)
                mode = "L"
                arr = slice_tensor
        else:
            mode = "L"
            arr = slice_tensor  # (H, W)

        arr = arr.float().detach().cpu()
        if per_slice_minmax:
            mn = arr.min()
            mx = arr.max()
            if (mx - mn) > 0:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = torch.zeros_like(arr)
        else:
            arr = arr.clamp(0.0, 1.0)

        arr = (arr * 255.0).round().to(torch.uint8).numpy()

        if mode == "RGB":
            # Convert (C, H, W) -> (H, W, C)
            arr = np.moveaxis(arr, 0, -1)

        return Image.fromarray(arr, mode=mode)
    
    if not batched:
        # (C, D, H, W) -> pick first channel if C>1
        if C == 1:
            slices = v[0, indices, :, :]  # (S, H, W)
            pil_list = [tensor_slice_to_pil(slices[i]) for i in range(num_slices)]
        else:
            # Keep all channels as RGB if C>=3; otherwise reduce to grayscale
            slices = v[:, indices, :, :]  # (C, S, H, W)
            pil_list = [tensor_slice_to_pil(slices[:, i]) for i in range(num_slices)]
        return pil_list
    else:
        # (B, C, D, H, W)
        pil_batches: List[List[Image.Image]] = []
        for b in range(v.shape[0]):
            if C == 1:
                slices = v[b, 0, indices, :, :]  # (S, H, W)
                pil_list = [tensor_slice_to_pil(slices[i]) for i in range(num_slices)]
            else:
                slices = v[b, :, indices, :, :]  # (C, S, H, W)
                pil_list = [tensor_slice_to_pil(slices[:, i]) for i in range(num_slices)]
            pil_batches.append(pil_list)
        return pil_batches



def sample_slices_from_volume_to_pil_fast(
    volume_tensor: torch.Tensor,
    num_slices: int = 4,
    random: bool = True,
    seed: Optional[int] = None,
    per_slice_minmax: bool = False,
) -> Union[List[Image.Image], List[List[Image.Image]]]:
    """
    Samples slices from a 3D volume and returns them as PIL Images (Optimized, Vectorized Version).

    Args:
        volume_tensor: 4D (C, D, H, W) or 5D (B, C, D, H, W) tensor.
        num_slices: Number of slices to sample.
        random: If True, randomly sample; otherwise, uniformly sample across depth.
        seed: Optional RNG seed for reproducible random sampling.
        per_slice_minmax: If True, min-max normalize each slice before converting.
                          If False, assume values are in [0, 1] and clip.

    Returns:
        - If input is 4D: List[PIL.Image.Image] of length num_slices.
        - If input is 5D: List[List[PIL.Image.Image]] with shape (B, num_slices).
    """
    # --- 1. Input Validation and Unified Path ---
    v = volume_tensor.float()  # Ensure float for calculations
    was_4d = False
    if v.ndim == 4:
        # (C, D, H, W) -> (1, C, D, H, W)
        v = v.unsqueeze(0)
        was_4d = True
    elif v.ndim != 5:
        raise ValueError(f"Expected 4D (C,D,H,W) or 5D (B,C,D,H,W) tensor, got shape {volume_tensor.shape}")

    B, C, D, H, W = v.shape
    device = v.device

    # --- 2. Generate Slice Indices ---
    if seed is not None and random:
        g = torch.Generator(device=device).manual_seed(seed)
    else:
        g = None

    if D <= num_slices:
        indices = torch.arange(D, device=device)
        if D < num_slices: # Pad by repeating the last slice index
            pad = indices[-1].repeat(num_slices - D)
            indices = torch.cat([indices, pad])
    else:
        if random:
            perm = torch.randperm(D, generator=g, device=device)
            indices = perm[:num_slices].sort().values
        else: # Uniformly spaced indices
            indices = torch.linspace(0, D - 1, steps=num_slices, device=device).round().long()

    # --- 3. Vectorized Slice Extraction ---
    # v is (B, C, D, H, W), indices is (S,)
    # We select along dimension 2 (Depth)
    slices = torch.index_select(v, dim=2, index=indices)  # (B, C, S, H, W)
    slices = slices.permute(0, 2, 1, 3, 4)  # (B, S, C, H, W) for easier processing

    # --- 4. Vectorized Channel and Normalization Logic ---
    # Reshape for easier per-slice operations
    slices = slices.reshape(B * num_slices, C, H, W)

    if C > 3 or C == 2: # Reduce to grayscale if not 1 or 3 channels
        slices = slices.mean(dim=1, keepdim=True) # (B*S, 1, H, W)
        C = 1
        mode = "L"
    elif C == 1:
        mode = "L"
    else: # C == 3
        mode = "RGB"

    if per_slice_minmax:
        # Vectorized min-max normalization per slice
        slice_min = slices.amin(dim=(-2, -1), keepdim=True)
        slice_max = slices.amax(dim=(-2, -1), keepdim=True)
        scale = slice_max - slice_min
        # Use torch.where to avoid division by zero
        slices = torch.where(scale > 1e-6, (slices - slice_min) / scale, torch.zeros_like(slices))
    else:
        slices = slices.clamp(0.0, 1.0)

    # --- 5. Vectorized Type Conversion and CPU Transfer ---
    # Scale to [0, 255] and convert to uint8 in one go
    slices = (slices * 255.0).round().to(torch.uint8)

    # Permute channels to the end for PIL (H, W, C) for RGB
    # And remove channel dim for grayscale (H, W)
    if mode == "RGB":
        slices = slices.permute(0, 2, 3, 1) # (B*S, H, W, 3)
    else: # mode == "L"
        slices = slices.squeeze(1) # (B*S, H, W)

    # Single, bulk transfer to CPU and convert to NumPy
    np_slices = slices.cpu().numpy()
    np_slices = np_slices.reshape(B, num_slices, *np_slices.shape[1:]) # (B, S, H, W, [C])

    # --- 6. Final (Unavoidable) Loop to Create PIL Images ---
    pil_batches = [
        [Image.fromarray(np_slices[b, s], mode=mode) for s in range(num_slices)]
        for b in range(B)
    ]

    # --- 7. Return in Original Format ---
    if was_4d:
        return pil_batches[0]
    else:
        return pil_batches


def sample_slices_from_volume_to_image(
    volume_tensor: torch.Tensor,
    num_slices: int = 4,
    random: bool = True,
    seed: Optional[int] = None,
    per_slice_minmax: bool = False,
    output_type: Literal["numpy", "torch"] = "numpy",
    channels: Literal["auto", "rgb", "grayscale"] = "auto",
    data_format: Literal["channels_last", "channels_first"] = "channels_last",
    dtype: Optional[Literal["uint8", "float32"]] = None,
) -> Union[
    List[np.ndarray], List[List[np.ndarray]],
    List[torch.Tensor], List[List[torch.Tensor]]
]:
    """
    Sample 'num_slices' along the depth (D) of a 3D volume and return images as NumPy or torch.

    Input:
        volume_tensor: (C, D, H, W) or (B, C, D, H, W).
        num_slices: number of slices to sample.
        random: random or uniform sampling across D.
        seed: RNG seed for reproducible random sampling.
        per_slice_minmax: normalize each slice to [0,1] via per-slice min-max.
        output_type: "numpy" or "torch".
        channels: "auto" (RGB if C>=3 else grayscale), "rgb", or "grayscale".
        data_format: "channels_last" (H, W, C) or "channels_first" (C, H, W).
        dtype: "uint8" (0..255) or "float32" ([0,1]). Defaults to uint8 for numpy,
               and uint8 for torch unless per_slice_minmax=True (then float32).

    Returns:
        - If input is (C, D, H, W): List[images] of length num_slices.
        - If input is (B, C, D, H, W): List[List[images]] length B, each inner list length num_slices.
          Each image is np.ndarray or torch.Tensor with shape determined by channels and data_format.
    """
    if volume_tensor.ndim == 4:
        C, D, H, W = volume_tensor.shape
        batched = False
    elif volume_tensor.ndim == 5:
        B, C, D, H, W = volume_tensor.shape
        batched = True
    else:
        raise ValueError(f"Expected 4D (C,D,H,W) or 5D (B,C,D,H,W), got {tuple(volume_tensor.shape)}")

    device = volume_tensor.device

    # Make indices
    def make_indices(D: int) -> torch.Tensor:
        if D <= num_slices:
            idx = torch.arange(D)
            if D < num_slices:
                idx = torch.cat([idx, idx[-1].repeat(num_slices - D)])
        else:
            if random:
                gen = torch.Generator(device="cpu")
                if seed is not None:
                    gen.manual_seed(seed)
                idx = torch.randperm(D, generator=gen)[:num_slices]
                idx, _ = torch.sort(idx)  # keep ascending order; remove for tiny extra speed
            else:
                idx = torch.linspace(0, D - 1, steps=num_slices).round().long().clamp(0, D - 1)
        return idx.to(device)

    idx = make_indices(D)

    def prepare_slices(v: torch.Tensor) -> torch.Tensor:
        """
        v: (C, D, H, W)
        Returns (S, H, W) for grayscale or (S, 3, H, W) for RGB in channels-first temporary format.
        """
        nonlocal channels
        if channels == "auto":
            channels = "rgb" if v.shape[0] >= 3 else "grayscale"

        if channels == "rgb":
            if v.shape[0] >= 3:
                gathered = v[:3].index_select(1, idx)        # (3, S, H, W)
                slices = gathered.permute(1, 0, 2, 3).contiguous()  # (S, 3, H, W)
            else:
                # Repeat single channel to 3
                single = v[0].index_select(0, idx)           # (S, H, W)
                slices = single.unsqueeze(1).repeat(1, 3, 1, 1)  # (S, 3, H, W)
        else:  # grayscale
            if v.shape[0] == 1:
                slices = v[0].index_select(0, idx)           # (S, H, W)
            else:
                gathered = v.index_select(1, idx)             # (C, S, H, W)
                slices = gathered.mean(dim=0)                 # (S, H, W)
        return slices

    def normalize_to_unit(slices: torch.Tensor, is_rgb: bool) -> torch.Tensor:
        """
        Convert to float32 in [0,1].
        - If per_slice_minmax: scale each slice by its min/max (over all channels for RGB).
        - Else: try to auto-detect range (if max>1, divide by 255).
        """
        if per_slice_minmax:
            slices = slices.to(torch.float32)
            if is_rgb:
                mn = slices.amin(dim=(1, 2, 3), keepdim=True)
                mx = slices.amax(dim=(1, 2, 3), keepdim=True)
            else:
                mn = slices.amin(dim=(1, 2), keepdim=True)
                mx = slices.amax(dim=(1, 2), keepdim=True)
            denom = (mx - mn).clamp_min(1e-6)
            return (slices - mn) / denom
        else:
            # Auto range: if integer or max>1.0, assume uint8 and divide by 255
            if slices.dtype.is_floating_point:
                # For safety, if values look >1, scale down
                if torch.isfinite(slices).all():
                    if slices.max() > 1.0:
                        slices = slices / 255.0
                slices = slices.clamp_(0.0, 1.0)
                return slices
            else:
                slices = slices.to(torch.float32) / 255.0
                return slices.clamp_(0.0, 1.0)

    def format_and_cast(slices: torch.Tensor, is_rgb: bool):
        """
        Format to channels_last or channels_first, then cast to requested dtype and output_type.
        """
        # Decide defaults
        nonlocal dtype, output_type, data_format
        if dtype is None:
            dtype = "uint8" if output_type == "numpy" else ("float32" if per_slice_minmax else "uint8")

        # Ensure float [0,1] before uint8 conversion
        if dtype == "uint8":
            slices = normalize_to_unit(slices, is_rgb=True if is_rgb else False)
            slices = slices.mul_(255.0).add_(0.5).to(torch.uint8)
        else:
            slices = normalize_to_unit(slices, is_rgb=True if is_rgb else False).to(torch.float32)

        # Format
        if is_rgb:
            if data_format == "channels_last":
                slices = slices.permute(0, 2, 3, 1).contiguous()  # (S, H, W, 3)
            else:
                # already (S, 3, H, W)
                pass
        else:
            # grayscale stays (S, H, W)
            # If you need 1-channel explicitly, uncomment:
            # if data_format == "channels_first": slices = slices.unsqueeze(1)  # (S, 1, H, W)
            pass

        # To NumPy or torch list
        if output_type == "numpy":
            arr = slices.detach().cpu().numpy()
            return [arr[i] for i in range(arr.shape[0])]
        else:
            # Keep on original device
            return [slices[i].contiguous() for i in range(slices.shape[0])]

    def process_item(v: torch.Tensor):
        slices = prepare_slices(v)  # (S, 3, H, W) or (S, H, W)
        is_rgb = (slices.ndim == 4)
        return format_and_cast(slices, is_rgb)

    if not batched:
        return process_item(volume_tensor)
    else:
        out: List[List[Union[np.ndarray, torch.Tensor]]] = []
        for b in range(volume_tensor.shape[0]):
            out.append(process_item(volume_tensor[b]))
        return out



def make_dummy_pil_slices(
    volume_tensor: torch.Tensor,
    num_slices: int = 4,
    mode: Optional[str] = None,  # "L" or "RGB"; None => auto (RGB if C>=3 else L)
    fill: int = 0,  # 0..255
    share_image_instance: bool = True,  # if True, reuse the same PIL object for all slots
) -> Union[List[Image.Image], List[List[Image.Image]]]:
    """
    Create num_slices zero-filled PIL Images matching the spatial size of volume_tensor.

    Args:
        volume_tensor: 4D (C, D, H, W) or 5D (B, C, D, H, W) tensor. Only H, W, and C are used.
        num_slices: number of dummy images to return per item.
        mode: "L" (grayscale) or "RGB". If None, picks RGB if C>=3 else L.
        fill: grayscale or RGB fill value (0..255). For RGB, a single int is broadcast to all channels.
        share_image_instance: reuse a single PIL Image object in all returned positions for speed.

    Returns:
        - If input is (C, D, H, W): List[PIL.Image.Image] of length num_slices.
        - If input is (B, C, D, H, W): List[List[PIL.Image.Image]] of length B, each inner list length num_slices.
    """
    # Validate dims and get H, W, C, B if present
    if volume_tensor.ndim == 4:
        C, D, H, W = volume_tensor.shape
        B = None
    elif volume_tensor.ndim == 5:
        B, C, D, H, W = volume_tensor.shape
    else:
        raise ValueError(f"Expected 4D (C,D,H,W) or 5D (B,C,D,H,W), got {tuple(volume_tensor.shape)}")

    # Decide mode automatically if not provided
    if mode is None:
        mode = "RGB" if C >= 3 else "L"

    # Build a single zero array and a single PIL Image
    if mode == "RGB":
        arr = np.empty((H, W, 3), dtype=np.uint8)
        arr.fill(int(fill) & 0xFF)
    elif mode == "L":
        arr = np.empty((H, W), dtype=np.uint8)
        arr.fill(int(fill) & 0xFF)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'L' or 'RGB'.")

    img = Image.fromarray(arr, mode=mode)

    # Build per-item slice lists
    if share_image_instance:
        slices = [img] * num_slices
    else:
        # Create distinct PIL objects (slower but safer if downstream mutates images)
        slices = [Image.fromarray(arr, mode=mode) for _ in range(num_slices)]

    # Return according to input rank
    if volume_tensor.ndim == 4:
        return slices
    else:
        # Create B independent lists (but each list may share the same image instance if share_image_instance=True)
        return [list(slices) for _ in range(B)]



# Example usage
if __name__ == "__main__":
    # Example of how to use the preprocessing functions
    print("Medical Image Preprocessing Module")
    print("=================================")

    if MONAI_AVAILABLE:
        print(f"MONAI version: {monai.__version__}")

        # Example preprocessing
        # preprocessor = MedicalImagePreprocessor()
        # tensor = preprocessor.preprocess_single_image("path/to/ct_scan.nii.gz")
        # print(f"Preprocessed tensor shape: {tensor.shape}")
    else:
        print("MONAI is not available. Please install it to use this module.")
