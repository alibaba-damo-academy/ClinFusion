import re
from typing import Optional
from pathlib import Path
import open_clip
from transformers import (
    AutoProcessor,
)
import numpy as np
from PIL import Image
import os



### V3: supoort multi-enc (dinov2, convnext, medsiglip) processors init
def load_new_processors(model_args, init_kwargs):
    new_enc_type = [t for t in [model_args.new_encoder_type, model_args.new_encoder_2_type, model_args.new_encoder_3_type] if t is not None]
    new_enc_path = [t for t in [model_args.new_encoder_path, model_args.new_encoder_2_path, model_args.new_encoder_3_path] if t is not None]
    assert len(new_enc_type) == len(new_enc_path)
    num_valid_processors = 0
    new_processors = {}
    new_processor_kwargs = {}
    # containing content like:
    #     "new_processor_1" = processor,
    #     "new_processor_1_type" = "dinov2",
    #     "new_processor_2" = processor,
    #     "new_processor_2_type" = "convnext",


    for enc_type, enc_path in zip(new_enc_type, new_enc_path):
        if "dinov2" in enc_type or "medsiglip" in enc_type:
            num_valid_processors += 1

            new_processor_1 = AutoProcessor.from_pretrained(
                enc_path,
                use_fast = model_args.use_fast_tokenizer,
                **init_kwargs,
            )
            new_processors[f"new_processor_{num_valid_processors}"]      = new_processor_1 if "dinov2" in enc_type else \
                                                                           new_processor_1.image_processor
            new_processors[f"new_processor_{num_valid_processors}_type"] = enc_type

        elif "convnext" in enc_type:
            num_valid_processors += 1

            # force_image_size = int(model_args.new_encoder_type.split("_")[-1]) # "convnext_768 -> 768"
            assert model_args.convnext_image_size is not None
            force_image_size = model_args.convnext_image_size
            model, transform_train, transform_val = open_clip.create_model_and_transforms(
                model_name = "convnext_large_d_320", # "convnext_xxlarge",
                pretrained = str(Path(enc_path) / "open_clip_pytorch_model.bin"),
                force_image_size=(force_image_size, force_image_size),
            ) # This model is abandoned since the open_clip does not have a function to create trasnforms only.
            new_processor_1 = transform_val

            new_processors[f"new_processor_{num_valid_processors}"]      = new_processor_1
            new_processors[f"new_processor_{num_valid_processors}_type"] = enc_type
        
        elif "merlin" in enc_type or "pe_3d" in enc_type:
            num_valid_processors += 1
            
            new_processors[f"new_processor_{num_valid_processors}"]      = None     # CT monai preprocessing is done in Qwen2VLPlugin in llamafactory/data/mm_plugin.py.
            new_processors[f"new_processor_{num_valid_processors}_type"] = enc_type

            assert (model_args.merlin_patch_size is not None) and (model_args.merlin_image_size is not None) and (model_args.merlin_num_channels is not None) and (model_args.merlin_volume_depth is not None)
            new_processor_kwargs["merlin_patch_size"]   = model_args.merlin_patch_size
            new_processor_kwargs["merlin_image_size"]   = model_args.merlin_image_size
            new_processor_kwargs["merlin_num_channels"] = model_args.merlin_num_channels
            new_processor_kwargs["merlin_volume_depth"] = model_args.merlin_volume_depth
            new_processor_kwargs["merlin_sample_images"] = model_args.merlin_sample_images

            ### My cache path
            if getattr(model_args, "CT_volume_cache", None) is not None:
                new_processor_kwargs["CT_volume_cache"]     = os.path.join(model_args.CT_volume_cache, \
                    f"depth_{model_args.merlin_volume_depth}_size_{model_args.merlin_image_size}_channel_{model_args.merlin_num_channels}")
            ### Yizeng cache path
            # new_processor_kwargs["CT_volume_cache"]     = model_args.CT_volume_cache

            CT_volume_cache_path = Path(new_processor_kwargs["CT_volume_cache"])
            CT_volume_cache_path.mkdir(parents=True, exist_ok=True)
            print("CT_volume_cache_path: ", CT_volume_cache_path)

        else:
            assert False, f"{enc_type} is not supported yet."
    
    return new_processors, new_processor_kwargs


def parse_cascade_mode(fusion_type: str) -> Optional[int]:
    """
    Parses a string like "cross_attn_cascade1" to find the cascade mode.

    It looks for the pattern "cascade<N>" and returns N as an integer.

    Args:
        fusion_type: The input string to parse.

    Returns:
        The integer value of the cascade mode, or None if the
        pattern is not found.
    """
    # The pattern looks for:
    # - "cascaed"
    # - (\d+): A capturing group for one or more digits (e.g., '1', '2')
    pattern = r"cascade(\d+)"

    match = re.search(pattern, fusion_type)

    if match:
        # match.group(1) contains the string from the first capturing group "(\d+)"
        return int(match.group(1))

    # Return None if no match was found.
    return None


def parse_expansion_factor(fusion_type: str) -> Optional[int]:
    """
    Parses a string like "cross_attn_expansion8" to find the expansion factor.

    It looks for the pattern "expansion<N>" and returns N as an integer.

    Args:
        fusion_type: The input string to parse.

    Returns:
        The integer value of the expansion factor, or None if the
        pattern is not found.
    """
    # The pattern looks for:
    # - "expansion"
    # - (\d+): A capturing group for one or more digits (e.g., '8', '16')
    pattern = r"expansion(\d+)"

    match = re.search(pattern, fusion_type)

    if match:
        # match.group(1) contains the string from the first capturing group "(\d+)"
        return int(match.group(1))

    # Return None if no match was found.
    return None


def parse_local_field(fusion_type: str) -> Optional[int]:
    """
    Parses a string like "cross_attn_local3x3" to find the local field size.

    It looks for a pattern like "local<N>x<M>" and returns N as an integer.

    Args:
        fusion_type: The input string to parse.

    Returns:
        The integer value of the local field size, or None if the
        pattern is not found.
    """
    # The pattern looks for:
    # - "local"
    # - (\d+): A capturing group for one or more digits (e.g., '3', '10')
    # - "x": The literal character 'x'
    # - \d+: One or more digits following the 'x'
    pattern = r"local(\d+)x\d+"

    match = re.search(pattern, fusion_type)

    if match:
        # match.group(1) contains the string from the first capturing group "(\d+)"
        return int(match.group(1))

    # Return None if no match was found, which is cleaner than raising an error.
    return None

def parse_use_ffn(fusion_type: str) -> bool:
    """
    Returns False if "ffnx" is in the string, True otherwise:
        if this string does not contain "ffn" or
        if this string has "ffn"
    """
    return "ffnx" not in fusion_type


def pil_to_rgb(img: Image.Image) -> Image.Image:
    """
    Converts any PIL image to a 3-channel RGB format.
    """
    # Already RGB
    if img.mode == "RGB":
        return img

    # Common modes that PIL can convert directly
    if img.mode in ("L", "P", "CMYK", "YCbCr", "HSV", "LAB", "RGBA", "LA", "1"):
        return img.convert("RGB")

    # 16-bit or float images: rescale to uint8 then convert
    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        # Convert to a float32 numpy array
        arr = np.array(img, dtype=np.float32)
        
        # Get min and max pixel values
        mn = float(arr.min())
        mx = float(arr.max())
        
        # Normalize to 0-1 range
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
            
        # Scale to 0-255 and convert to uint8
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        
        # Create a grayscale PIL image from the array and then convert to RGB
        return Image.fromarray(arr, mode="L").convert("RGB")

    # Fallback for any other unhandled modes
    return img.convert("RGB")
