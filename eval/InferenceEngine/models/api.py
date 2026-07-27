import os
import base64
import io
import asyncio
import json
from typing import List, Dict, Any
import nibabel as nib
from .utils import read_image, read_nifti, image_resize, nifti_to_image_slices

# Import specific error types from the openai library
from openai import AsyncOpenAI, APIStatusError, APIError
import PIL.Image
import random

class APIWrapper:
    """
    A generic API wrapper that conforms to the OpenAI standard and fits into the
    MedEvalKit-V3 framework. It uses asyncio for concurrent API requests.
    """
    def __init__(self, model_path: str, model_config: dict, generation_config: dict):
        """
        Initializes the AsyncOpenAI client from the flattened model_config.
        
        Args:
            model_path (str): For API models, this is repurposed to be the model name
                              (e.g., "gpt-4-vision-preview").
            model_config (dict): A dictionary containing API settings like base_url, etc.
            generation_config (dict): A dictionary with temperature, max_tokens, etc.
        """
        print("[APIWrapper] Initializing for API-based inference...")

        # The model name is now passed via the `model_path` argument.
        self.model_name = model_path
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please provide your API key.")
            
        # Read API settings directly from the top level of model_config.
        base_url = model_config.get('base_url')

        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.generation_config = generation_config
        
        # Read retry logic settings directly from model_config, with defaults.
        self.max_retries = model_config.get('max_retries', 10)
        self.min_image_size = model_config.get('min_image_size', 32)
        self.max_image_size = model_config.get('max_image_size', 1024)
        self.max_image_num = model_config.get('max_image_num', 10)
        
        print(f"[APIWrapper] Ready to send requests to model: {self.model_name}")

    def _encode_image_to_base64(self, image: PIL.Image.Image, format="JPEG") -> str:
        """Encodes a PIL.Image object to a base64 string."""
        buffered = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _prepare_api_request(self, item: Dict[str, Any]) -> dict:
        """Prepares the payload for a single API request, including image and NIfTI data."""
        messages = item.get("messages", {})
        prompt_text = messages.get("prompt")
        images = messages.get("image", [])
        niftis = messages.get("nifti", []) # ### UPGRADE: Get nifti data

        user_content = [{"type": "text", "text": prompt_text}]
        image_count = 0

        # --- 1. Process standard images first ---
        for image in images:
            base64_image = self._encode_image_to_base64(image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            image_count += 1

        if niftis:
            for nifti_idx, nifti_obj in enumerate(niftis):
                if image_count >= self.max_image_num:
                    print(f"Warning: Reached max image limit ({self.max_image_num}). Skipping remaining NIfTI files.")
                    break

                # --- Dynamic Slice Calculation ---
                available_slots = self.max_image_num - image_count
                remaining_niftis = len(niftis) - nifti_idx
                num_slices_to_extract = max(1, available_slots // remaining_niftis)

                image_slices = nifti_to_image_slices(nifti_obj, num_slices=num_slices_to_extract)
                
                for slice_img in image_slices:
                    if image_count >= self.max_image_num:
                        break
                    
                    base64_slice = self._encode_image_to_base64(slice_img)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_slice}"}
                    })
                    image_count += 1

        api_messages = [{"role": "user", "content": user_content}]
        if "system" in messages and messages["system"]:
            api_messages.insert(0, {"role": "system", "content": messages["system"]})

        return {
            "model": self.model_name,
            "messages": api_messages,
            "max_tokens": self.generation_config['max_new_tokens'],
            "temperature": self.generation_config['temperature'],
            "top_p": self.generation_config['top_p'],
        }

    async def _generate_with_retry(self, payload: dict):
        """
        Makes a single API call with robust retry logic.
        Retries on all errors with a random delay, but has special logic for truncated responses.
        """
        # This counter is for all general errors (rate limits, server errors, etc.)
        error_retry_attempt = 0
    
        current_payload = payload.copy()

        while True:
            try:
                # Make the API call
                completion = await self.async_client.chat.completions.create(**current_payload)
                content = completion.choices[0].message.content if completion.choices else None

                # Check for implicit rate limits (e.g., a 200 OK response with an error message)
                # If found, raise an error to be caught by the generic exception handler below.
                if content is None or ('rate limit' in content.lower() or '平台限流' in content):
                    raise APIError(message="Implicit rate limit detected in response body.", response=None, body=content)

                # Handle response truncation due to 'length' finish_reason
                # This has its own separate retry logic and counter.
                if completion.choices[0].finish_reason == 'length':
                    if error_retry_attempt >= self.max_retries:
                        print("Warning: Response still truncated after all length-based retries. Returning last response.")
                        return completion # Give up and return the truncated response
                    
                    # Increase the max_tokens for the next attempt
                    old_tokens = current_payload.get('max_tokens', 1024)
                    new_tokens = min(old_tokens * 2, 16384) # Double tokens, but cap at a reasonable limit
                    current_payload['max_tokens'] = new_tokens
                    error_retry_attempt += 1
                    
                    print(f"Warning: Response truncated. Retrying with max_tokens={new_tokens}. (Attempt {error_retry_attempt}/{self.max_retries})")
                    continue # Immediately retry with the new payload
                
                # If we get here, the call was successful and not truncated.
                return completion

            except Exception as e:
                # This is the unified handler for ALL other exceptions.
                if error_retry_attempt < self.max_retries:
                    # Use a random delay instead of exponential backoff
                    delay = random.uniform(1.0, 5.0) 
                    
                    error_retry_attempt += 1
                    print(f"An error occurred: {type(e).__name__}. Retrying in {delay:.2f}s... (Attempt {error_retry_attempt}/{self.max_retries})")
                    
                    await asyncio.sleep(delay)
                    continue # Go to the next iteration of the while loop to retry the API call
                else:
                    # If max retries are exhausted, log and re-raise the exception
                    print(f"API call failed after {self.max_retries} retries due to error: {e}")
                    raise e

    async def _generate_concurrently(self, batch_data: List[Dict[str, Any]]) -> list:
        """Sends all API requests concurrently."""
        # The number of concurrent requests is limited by the batch_size from config
        tasks = [self._generate_with_retry(self._prepare_api_request(item)) for item in batch_data]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def generate(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """
        The main entry point for the wrapper, matching the local model interface.
        """
        completions = asyncio.run(self._generate_concurrently(batch_data))
        
        output_texts = []
        for result in completions:
            if isinstance(result, Exception):
                error_msg = f"Error: API call failed after all retries. Details: {str(result)}"
                print(error_msg)
                output_texts.append(error_msg)
            else:
                output_texts.append(result.choices[0].message.content.strip())
        return output_texts