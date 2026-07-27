import PIL.Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from typing import List, Dict, Any
import nibabel as nib
from .utils import nifti_to_image_slices

class QwenVLWrapper:
    """
    A wrapper for Qwen-VL models using vLLM for efficient, batched inference.
    This class handles prompt formatting, image processing, and generation.
    """
    def __init__(self, model_path: str, model_config: dict, generation_config: dict):
        print(f"[QwenVLWrapper] Initializing model from: {model_path}")
        
        # 1. Initialize the vLLM engine
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=model_config.get('gpus_per_worker', 1),
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"image": model_config.get('max_image_num', 10)},
            distributed_executor_backend=model_config.get('distributed_executor_backend', "mp"),
            max_model_len=model_config.get('max_model_len', 8192),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.95),
            mm_processor_cache_gb=model_config.get('mm_processor_cache_gb', 50)
        )
        
        # 2. Initialize the processor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        # 3. Store generation parameters
        self.sampling_params = SamplingParams(
            temperature=generation_config.get('temperature', 0.0),
            top_p=generation_config.get('top_p', 1.0),
            max_tokens=generation_config.get('max_tokens', 4096),
            stop_token_ids=[self.tokenizer.eos_token_id],
            # Add other params like repetition_penalty if needed
        )

        self.min_image_size = model_config.get('min_image_size', 32)
        self.max_image_size = model_config.get('max_image_size', 1024)
        self.max_image_num = model_config.get('max_image_num', 10)
        print("[QwenVLWrapper] Model initialized successfully.")

    def generate(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """
        Processes a batch of data and generates text responses.
        This version is optimized to handle pre-loaded PIL.Image objects,
        image paths, and NIfTI files (paths or nibabel objects).
        """
        mm_requests = []
        text_only_prompts = []
        original_indices_map = []

        for i, item in enumerate(batch_data):
            messages = item.get("messages", {})
            prompt_text = messages.get("prompt", "")
            images = messages.get("image", [])
            niftis = messages.get("nifti", [])

            chat_content = []
            user_content = []
            images_for_prompt = []

            # --- 1. Process standard images first ---
            # images的数量在rollout.py已经限制住了
            if images:
                for img in images:
                    user_content.append({"type": "image"})
                    images_for_prompt.append(img)

            # 由于这里nifti是通过image来输入，所以这里要手动限制住slice数量
            if niftis:
                for nifti_idx, nifti_obj  in enumerate(niftis):
                    # Check if we have any capacity left for more images
                    if len(images_for_prompt) >= self.max_image_num:
                        print(f"Warning: Reached max image limit ({self.max_image_num}). Skipping remaining NIfTI files.")
                        break
                    
                    available_slots = self.max_image_num - len(images_for_prompt)
                    remaining_niftis = len(niftis) - nifti_idx
                    # Distribute available slots among remaining NIfTIs, ensuring at least 1 slice
                    num_slices_to_extract = max(1, available_slots // remaining_niftis)

                    # Convert NIfTI to a list of PIL Images, uniformly
                    image_slices = nifti_to_image_slices(nifti_obj, num_slices=num_slices_to_extract)
    
                    # Add each slice to the prompt until capacity is full
                    for slice_img in image_slices:
                        if len(images_for_prompt) >= self.max_image_num:
                            break # Stop if we hit the limit mid-way through slices
                        
                        # Resize the slice just like a regular image
                        user_content.append({"type": "image"})
                        images_for_prompt.append(slice_img)
            
            # --- 2. Construct the final prompt ---
            user_content.append({"type": "text", "text": prompt_text})
            chat_content.append({"role": "user", "content": user_content})

            prompt_str = self.tokenizer.apply_chat_template(
                chat_content, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # --- 3. Route to the correct vLLM queue (multi-modal or text-only) ---
            if images_for_prompt:
                mm_data = {"image": images_for_prompt}
                mm_requests.append({
                    "prompt": prompt_str,
                    "multi_modal_data": mm_data,
                })
                original_indices_map.append({'type': 'mm', 'index': len(mm_requests) - 1})
            else:
                text_only_prompts.append(prompt_str)
                original_indices_map.append({'type': 'text', 'index': len(text_only_prompts) - 1})

        # --- 4. Perform generation ---
        mm_outputs = []
        text_only_outputs = []

        if mm_requests:
            mm_outputs = self.llm.generate(mm_requests, self.sampling_params)

        if text_only_prompts:
            text_only_outputs = self.llm.generate(text_only_prompts, self.sampling_params)

        # --- 5. Reassemble results in original order ---
        final_results = ["" for _ in range(len(batch_data))]
        for i, mapping in enumerate(original_indices_map):
            if mapping['type'] == 'mm':
                output = mm_outputs[mapping['index']]
                final_results[i] = output.outputs[0].text.strip()
            else: # type == 'text'
                output = text_only_outputs[mapping['index']]
                final_results[i] = output.outputs[0].text.strip()
                
        assert len(final_results) == len(batch_data), "Mismatch in final results length"

        return final_results
