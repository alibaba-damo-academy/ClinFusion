import os
import json
from pathlib import Path
from typing import List, Dict, Any, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import UnidentifiedImageError, Image
import nibabel as nib

# --- ADDED: Self-contained MONAI imports for the new preprocessing function ---
try:
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        Resized,
        ToTensord,
        EnsureChannelFirst,
        Orientation,
        Spacing,
        ScaleIntensityRange,
        Resize,
        ToTensor,
    )
    from monai.data import MetaTensor
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("WARNING: MONAI library not found. 3D volume processing will not be available.")
    # Define dummy classes if MONAI is not available to avoid runtime errors on import
    class Compose: pass
    class MetaTensor: pass

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, Qwen2_5_VLProcessor
from transformers.modeling_utils import load_sharded_checkpoint

# Custom imports from the same directory structure
from .modeling_multi_enc_qwen_vl import MultiEnc_Qwen2_5_VLForConditionalGeneration
from .multi_enc_utils import load_new_processors
from .loader import add_new_enc_to_model
from .image_processing_multi_enc_qwen2_vl import ImageMultiProcessor
# This import is still needed for the slice sampling function
from .medical_image_preprocessing import sample_slices_from_volume_to_pil_fast

# ==============================================================================
# 1. NEW SELF-CONTAINED PREPROCESSING FUNCTION
# ==============================================================================

def _ct_preprocessing_flexible(
    path_or_object: Union[str, nib.Nifti1Image],
    target_size: tuple = (64, 512, 512),
    target_spacing: tuple = (1.0, 1.0, 1.0),
    intensity_window: tuple = (-1000, 1000)
) -> torch.Tensor:
    """
    A flexible MONAI preprocessing function that handles both file paths and
    in-memory nibabel objects.

    Args:
        path_or_object: A file path (str) to a NIfTI file or a loaded nibabel.Nifti1Image object.
        target_size: The target output volume size (D, H, W).
        target_spacing: The target voxel spacing in mm.
        intensity_window: The Hounsfield Unit (HU) window for intensity scaling.

    Returns:
        A preprocessed torch.Tensor of shape (1, D, H, W).
    """
    if not MONAI_AVAILABLE:
        print("Error: MONAI is required for 3D preprocessing but is not installed.")
        # Return a zero tensor that matches the expected output shape
        return torch.zeros(1, *target_size, dtype=torch.float32)

    # --- Path 1: Input is a file path string ---
    if isinstance(path_or_object, str):
        # Use the dictionary-based transforms which are designed for file paths.
        transforms = Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
            ScaleIntensityRanged(keys=["image"], a_min=intensity_window[0], a_max=intensity_window[1], b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=["image"], spatial_size=target_size, mode="trilinear"),
            ToTensord(keys=["image"]),
        ])
        # The pipeline expects a dictionary
        data_dict = {"image": path_or_object}
        processed_data = transforms(data_dict)
        # The output tensor will have shape (C, D, H, W)
        one_volume = processed_data["image"]

    # --- Path 2: Input is an in-memory nibabel object ---
    elif isinstance(path_or_object, nib.Nifti1Image):
        # Use the array-based transforms for in-memory processing.
        transforms = Compose([
            # No loading needed, we start with the array
            EnsureChannelFirst(channel_dim="no_channel"),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=target_spacing, mode="bilinear"),
            ScaleIntensityRange(a_min=intensity_window[0], a_max=intensity_window[1], b_min=0.0, b_max=1.0, clip=True),
            Resize(spatial_size=target_size, mode="trilinear"),
            ToTensor(),
        ])
        # Get data and affine matrix from the nibabel object
        data_array = path_or_object.get_fdata(dtype=np.float32)
        affine = path_or_object.affine
        
        # Create a MetaTensor, which wraps the data and metadata (affine matrix).
        # MONAI transforms will automatically use this metadata.
        # Note: data_array may need a channel dimension added. MONAI expects (C, H, W, D).
        # We will let EnsureChannelFirst handle this.
        meta_tensor = MetaTensor(x=data_array, affine=affine)
        
        # Apply the pipeline directly to the MetaTensor
        one_volume = transforms(meta_tensor)

    else:
        raise TypeError(f"Unsupported input type for preprocessing: {type(path_or_object)}")

    # --- Final Formatting and Safety Check ---
    # The output should be (C, D, H, W). We only want the first channel.
    if one_volume.shape[0] > 1:
        one_volume = one_volume[0:1, ...] # Shape: (1, D, H, W)
    
    # Safety check to ensure final size is correct
    if one_volume.shape[-3:] != target_size:
        one_volume = F.interpolate(
            one_volume.unsqueeze(0), size=target_size, mode="trilinear", align_corners=False
        ).squeeze(0)

    return one_volume


# ==============================================================================
# 2. HELPER FUNCTIONS AND CLASSES (UNCHANGED)
# ==============================================================================

try:
    from vortex_io.storage import Blosc2NDArrStorage
except ImportError:
    print("WARNING: 'vortex_io' library not found. Caching functionality will be unavailable.")
    print("         A dummy Storage class will be used, which may reduce performance.")
    class Blosc2NDArrStorage:
        def load(self, path):
            raise FileNotFoundError("vortex_io not found, cannot load from cache.")
        def save(self, arr, path):
            pass # Do nothing
        def asarray_from_txyz(self, arr):
            return arr # Pass through

class DummyArgs:
    """A helper class to hold model configuration arguments, mimicking an 'args' object."""
    def __init__(self,
        new_encoder_type,
        new_encoder_path,
        new_encoder_2_type,
        new_encoder_2_path,
        compute_dtype,
        multi_enc_fusion_type,
        convnext_image_size,
        use_fast_tokenizer = True,
    ):
        self.new_encoder_type = new_encoder_type
        self.new_encoder_path = new_encoder_path
        self.new_encoder_2_type = new_encoder_2_type
        self.new_encoder_2_path = new_encoder_2_path
        self.compute_dtype = compute_dtype
        self.multi_enc_fusion_type = multi_enc_fusion_type
        self.convnext_image_size = convnext_image_size
        self.use_fast_tokenizer = use_fast_tokenizer

    def update(self, kwargs_dict: dict):
        """Updates attributes from a dictionary."""
        self.__dict__.update(kwargs_dict)

    def __repr__(self):
        """Provides a developer-friendly string representation of the object."""
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

def get_config_values(
    config_path: str,
    keys_to_extract: List[str],
    strict: bool = False
) -> Dict[str, Any]:
    """
    Reads a JSON config file and extracts specified top-level or nested keys.
    Nested keys can be specified using dot notation (e.g., 'vision_config.hidden_size').
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        config_file_in_dir = config_file.parent / "config.json"
        if (config_file.name != "config.json") and config_file_in_dir.is_file():
             config_file = config_file_in_dir
        else:
             raise FileNotFoundError(f"Config file not found at: {config_path}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from {config_file}: {e.msg}", e.doc, e.pos)

    extracted_values = {}
    for full_key in keys_to_extract:
        key_parts = full_key.split('.')
        current_level_data = config_data
        found = True
        for part in key_parts:
            if isinstance(current_level_data, dict) and part in current_level_data:
                current_level_data = current_level_data[part]
            else:
                found = False
                break

        if found:
            extracted_values[full_key] = current_level_data
        else:
            if strict:
                raise KeyError(f"Key '{full_key}' not found in the config file at '{config_file}'.")
            else:
                extracted_values[full_key] = None

    return extracted_values

# ==============================================================================
# 3. UNIFIED MedEvalKitAdapter CLASS DEFINITION (UPDATED)
# ==============================================================================

class MedEvalKitAdapter:
    """
    An adapter to make the complex 'Multi_Enc_Qwen2_5_VL_Legacy' model
    compatible with a standard inference interface.

    This single class now handles all model loading, pre-processing,
    and generation logic.
    """
    def __init__(self, model_path: str, model_config: dict, generation_config: dict):
        """
        Handles the complex initialization of the Multi-Enc-Qwen model.
        """
        print("[MedEvalKitAdapter] Initializing Multi-Enc-Qwen model...")

        # --- 1. Set up device and generation parameters ---
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.device = f"cuda:{local_rank}"
        self.temperature = generation_config.get("temperature", 0.0)
        self.top_p = generation_config.get("top_p", 1.0)
        self.repetition_penalty = generation_config.get("repetition_penalty", 1.0)
        self.max_new_tokens = generation_config.get("max_new_tokens", 1024)

        # --- 2. Initialize the Base LLM and add custom encoders ---
        original_qwen2_5_vl_path = "/mnt/workspace/workgroup/hangjie.yhj/code_medical/Lingshu-2/ckpts/models/Qwen2.5-VL-7B-Instruct"
        original_new_encoder_path = {
            "dinov2": "/mnt/workspace/workgroup/hangjie.yhj/code_medical/Lingshu-2/ckpts/models/dinov2-large",
            "convnext": "/mnt/workspace/workgroup/hangjie.yhj/code_medical/Lingshu-2/ckpts/models/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
            "medsiglip": "/mnt/workspace/workgroup/hangjie.yhj/code_medical/Lingshu-2/ckpts/models/medsiglip-448",
            "merlin": "",
        }

        self.llm = MultiEnc_Qwen2_5_VLForConditionalGeneration.from_pretrained(
            original_qwen2_5_vl_path,
            torch_dtype="auto",
            device_map={"": self.device},
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
        )

        config_values = get_config_values(
            model_path,
            [
                "vision_config.fusion_type",
                "vision_config.vision_model_config",
                "vision_config.vision_model_config.convnext.model_cfg.vision_cfg.image_size"
            ]
        )
        new_encoder_types = [one_type for one_type in config_values["vision_config.vision_model_config"].keys() if one_type != "original_enc"]

        model_args = DummyArgs(
            new_encoder_type=new_encoder_types[0],
            new_encoder_path=original_new_encoder_path[new_encoder_types[0]],
            new_encoder_2_type=None if len(new_encoder_types) <= 1 else new_encoder_types[1],
            new_encoder_2_path=None if len(new_encoder_types) <= 1 else original_new_encoder_path[new_encoder_types[1]],
            compute_dtype="auto",
            multi_enc_fusion_type=config_values["vision_config.fusion_type"],
            convnext_image_size=config_values["vision_config.vision_model_config.convnext.model_cfg.vision_cfg.image_size"]
        )

        self.llm = add_new_enc_to_model(self.llm, model_args)

        # --- 3. Load the fine-tuned model weights (sharded checkpoint) ---
        load_info = load_sharded_checkpoint(self.llm, model_path, strict=False, prefer_safe=True)
        print(f"Loaded model from checkpoint: {model_path}\nLoad info: {load_info}")
        self.llm.eval()

        # --- 4. Initialize the custom multi-modal processor ---
        init_kwargs = {'trust_remote_code': False, 'cache_dir': None, 'revision': 'main', 'token': None}
        processor = AutoProcessor.from_pretrained(original_qwen2_5_vl_path, **init_kwargs)

        additional_processor_values = get_config_values(
            os.path.join(model_path, "preprocessor_config.json"),
            [
                "merlin_patch_size", "merlin_image_size",
                "merlin_num_channels", "merlin_volume_depth"
            ]
        )
        additional_processor_values["CT_volume_cache"] = "/mnt/workspace/workgroup/hangjie.yhj/code_medical/MedEvalKit-V2-25-11-07/.cache/3d_volume"
        model_args.update(additional_processor_values)

        new_processors, new_processor_kwargs = load_new_processors(model_args, init_kwargs)
        image_multi_processor = ImageMultiProcessor(processor.image_processor, **new_processors, **new_processor_kwargs)
        processor.image_processor = image_multi_processor
        self.processor = processor

        self.processor.tokenizer.padding_side = 'left'
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.eos_token_id = self.processor.tokenizer.eos_token_id

        print("[MedEvalKitAdapter] Multi-Enc-Qwen model initialized successfully.")

    def generate(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """
        Receives standard batch data, adapts it, calls generation, and returns text results.
        """
        prompts = []
        all_images = []
        all_volumes = []

        for item in batch_data:
            messages = item.get("messages", {})

            assert "nifti" in messages, \
                'The "messages" dictionary must contain a "nifti" key.'

            user_prompt = messages.get("prompt", "")
            image_inputs = messages.get("image", [])
            nifti_inputs = messages.get("nifti", [])
            all_inputs = image_inputs + nifti_inputs
            
            volume, processed_image_inputs = self._pre_sample_slices_from_volume(all_inputs, self.processor)

            pil_images = [self._read_pil_image(img_input) for img_input in processed_image_inputs]
            pil_images = [img for img in pil_images if img is not None]

            content = []
            for _ in pil_images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": user_prompt})

            formatted_prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True
            )

            prompts.append(formatted_prompt)
            all_images.append(pil_images if pil_images else None)
            all_volumes.append(volume)

        inputs = self.processor(
            text=prompts,
            images=all_images,
            padding=True,
            return_tensors="pt",
            images_kwargs={"volume": all_volumes}
        ).to(self.device)

        input_ids_len = inputs.input_ids.shape[1]

        gen_kwargs = {
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            outputs = self.llm.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[:, input_ids_len:]
        output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return output_texts

    def _read_pil_image(self, image_input):
        """Safely reads an image from a path or returns it if already a PIL image."""
        try:
            if isinstance(image_input, Image.Image):
                return image_input
            if isinstance(image_input, (str, Path)):
                if not os.path.exists(image_input):
                    print(f"Error: Invalid file path -> {image_input}")
                    return None
                return Image.open(image_input).convert("RGB")
        except (UnidentifiedImageError, IOError, SyntaxError) as e:
            print(f"Error: Read image failed for '{image_input}'. {e}")
            return None

    def _pre_sample_slices_from_volume(self, inputs, processor):
        """
        Processes NIfTI files (from path or object) into volumes and sampled slices.
        Separates NIfTI inputs from regular image inputs.
        """
        volume_flag = any(
            (isinstance(p, str) and p.endswith((".nii", ".nii.gz", ".npy"))) or \
            isinstance(p, nib.Nifti1Image)
            for p in inputs
        )

        if not volume_flag:
            return None, inputs

        volume_tensors = []
        sampled_images = []
        remaining_image_inputs = []

        for item in inputs:
            one_volume = None
            if isinstance(item, str) and item.endswith((".nii", ".nii.gz", ".npy")):
                one_volume = self._cache_and_load_nii_gz(item, processor)
            elif isinstance(item, nib.Nifti1Image):
                one_volume = self._process_nifti_object(item, processor)
            else:
                remaining_image_inputs.append(item)
                continue

            if one_volume is not None:
                volume_tensors.append(one_volume)
                # This function is still imported from the other file
                new_images = sample_slices_from_volume_to_pil_fast(one_volume)
                sampled_images.extend(new_images)

        final_image_inputs = sampled_images + remaining_image_inputs
        return volume_tensors if volume_tensors else None, final_image_inputs

    def _cache_and_load_nii_gz(self, nii_path, processor):
        """Loads a NIfTI volume from cache or file, processing it to the required size."""
        cache_path = processor.image_processor.CT_volume_cache
        b2nd_cache_path = Path(cache_path) / (Path(nii_path).name.replace(".nii.gz", "").replace(".nii", "") + ".b2nd")

        if b2nd_cache_path.exists():
            one_volume = self._load_from_cache(b2nd_cache_path)
            if one_volume is not None:
                print(f"Read volume from cached local: {b2nd_cache_path}")
                return one_volume
        
        print(f"Processing volume from file: {nii_path}")
        target_depth = processor.image_processor.merlin_volume_depth
        target_image_size = processor.image_processor.merlin_image_size
        target_size = (target_depth, target_image_size, target_image_size)
        
        try:
            # --- UPDATED: Use the new flexible function ---
            one_volume = _ct_preprocessing_flexible(nii_path, target_size=target_size)
            print(f"Successfully processed volume from file: {nii_path}")
            
            # Cache the result
            try:
                storage_io = Blosc2NDArrStorage()
                blosc2_ndarr = storage_io.asarray_from_txyz(one_volume.contiguous().numpy())
                storage_io.save(blosc2_ndarr, b2nd_cache_path)
                print(f"Saved processed volume to cache: {b2nd_cache_path}")
            except Exception as e:
                print(f"Warning: Saving volume to cache failed: {e}")

        except Exception as e:
            print(f"Preprocessing failed for {nii_path}: {e}. Falling back.")
            one_volume = torch.zeros(1, *target_size, dtype=torch.float32)
            print("Warning: Failed to load volume. Using a zero tensor as a placeholder.")
        
        return one_volume

    def _process_nifti_object(self, nifti_object: nib.Nifti1Image, processor) -> torch.Tensor:
        """Processes a pre-loaded nibabel NIfTI object in-memory."""
        print("Processing pre-loaded NIfTI object in memory...")
        target_depth = processor.image_processor.merlin_volume_depth
        target_image_size = processor.image_processor.merlin_image_size
        target_size = (target_depth, target_image_size, target_image_size)

        one_volume = _ct_preprocessing_flexible(nifti_object, target_size=target_size)
        print("Successfully processed NIfTI object using MONAI.")

        return one_volume

    def _load_from_cache(self, cache_file: Path) -> torch.Tensor | None:
        """Loads a tensor from a blosc2 cache file."""
        storage_io = Blosc2NDArrStorage()
        try:
            loaded_image_np = storage_io.load(str(cache_file))
            return torch.from_numpy(loaded_image_np.copy())
        except Exception as e:
            print(f"Warning: Failed to read cache file {cache_file}: {e}. Will regenerate.")
            try:
                cache_file.unlink()
            except OSError as remove_e:
                print(f"Warning: Could not delete corrupted cache file: {remove_e}")
            return None
