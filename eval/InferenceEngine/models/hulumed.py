import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os
from typing import List, Dict, Any, Union
from .utils import read_image, read_nifti, image_resize
import nibabel as nib
import numpy as np

class HuluMedWrapper:
    """
    A wrapper for HuluMed-HF models, mimicking the QwenVLWrapper's design philosophy.
    This class handles explicit data loading before passing in-memory objects to the processor.
    """
    def __init__(self, model_path: str, model_config: dict=None, generation_config: dict = None):
        print(f"[HuluMedWrapper] Initializing model from: {model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        
        if generation_config is None:
            generation_config = {}
            
        self.max_new_tokens = generation_config.get("max_new_tokens", 1024)
        
        # Placeholder resizing parameters for API consistency, though HuluMed processor handles it.
        self.min_image_size = model_config.get('min_image_size', 32)
        self.max_image_size = model_config.get('max_image_size', 1024)
        self.max_image_num = model_config.get('max_image_num', 10)
        self.max_nifti_slices= model_config.get('max_nifti_slices', 180)
        
        print("[HuluMedWrapper] Model initialized successfully.")

    def generate(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """
        Generates text by correctly pre-processing all data into the exact in-memory
        formats that the processor's backend functions expect.
        """
        generated_texts = []
        for item in batch_data:
            messages = item.get("messages", {})
            
            images_for_processor = []
            prompt_content = []

            # --- 1. Intelligent Data Handling & Pre-processing ---
            # Handle Images
            if "image" in messages and messages["image"]:
                for img in messages["image"]:
                    prompt_content.append({"type": "image"})
                    images_for_processor.append(('image', img))
              
            # Handle 3D nifti files by SLICING them into 2D images
            if "nifti" in messages and messages["nifti"] and self.max_nifti_slices > 0:
                for nii_obj in messages["nifti"]:
                    volume = nii_obj.get_fdata()
                    slices_3d = [volume[:, :, i] for i in range(volume.shape[2])]
                    
                    if len(slices_3d) > self.max_nifti_slices:
                        indices = np.linspace(0, len(slices_3d) - 1, self.max_nifti_slices, dtype=int)
                        slices_3d = [slices_3d[i] for i in indices]

                    slice_images = []
                    for slice_2d in slices_3d:
                        slice_min, slice_max = slice_2d.min(), slice_2d.max()
                        if slice_max > slice_min:
                            slice_2d = (slice_2d - slice_min) / (slice_max - slice_min) * 255.0
                        slice_2d_uint8 = slice_2d.astype(np.uint8)
                        slice_images.append(Image.fromarray(slice_2d_uint8).convert("RGB"))
                    
                    # The placeholder dictionary MUST include 'num_frames' for the template's range() loop.
                    prompt_content.append({"type": "video", "num_frames": len(slice_images)})
                    images_for_processor.append(('video', slice_images))
                            
            
            # Add text prompt last
            prompt_content.append({"type": "text", "text": messages["prompt"]})
            conversation = [{"role": "user", "content": prompt_content}]
            
            # --- 2. Call the Processor Correctly ---
            inputs = self.processor(
                conversation=conversation,
                images=images_for_processor,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # --- 3. Process and Generate ---
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            
            response = self.processor.batch_decode(output_ids, skip_special_tokens=True, use_think=False)[0].strip()
            generated_texts.append(response)
            
        return generated_texts



