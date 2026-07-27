# First, ensure all necessary imports are available
# We will need to copy some utility functions and classes that the processors depend on.
# In a real project, you would import them from transformers.
# For this example, let's stub them out if they aren't critical.
import numpy as np
import PIL.Image
from typing import Optional, Union, List, Dict
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import (
    make_list_of_images,
)
from transformers.utils import logging
from transformers import AutoProcessor
import os
import json
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from .multi_enc_utils import pil_to_rgb

logger = logging.get_logger(__name__)

# NOTE: These are simplified stubs for the example to run. 
# In a real scenario, you would import these from transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl
def get_size_dict(*args, **kwargs):
    return {"height": 224, "width": 224}

def validate_preprocess_arguments(*args, **kwargs):
    pass

def smart_resize(*args, **kwargs):
    return 448, 448

def make_flat_list_of_images(images):
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], list):
        return [item for sublist in images for item in sublist]
    return make_list_of_images(images)

def make_batched_videos(videos):
    return videos


# ==============================================================================
# === NEW MASTER PREPROCESSOR ===
# ==============================================================================

### v2: support 2-encoder addition
class ImageMultiProcessor(BaseImageProcessor):
    """
    A master preprocessor that composes two different sub-preprocessors to process
    an image in two distinct ways.
    """
    # Define the inputs this processor will ultimately create
    model_input_names = ["pixel_values", "pixel_values_dinov2", "pixel_values_convnext", "pixel_values_medsiglip", "image_grid_thw"]

    def __init__(self, 
      original_processor = None, 
      new_processor_1 = None,
      new_processor_1_type = None,
      new_processor_2 = None,
      new_processor_2_type = None,
      new_processor_3 = None,
      new_processor_3_type = None,
      **kwargs
    ):
        super().__init__(**kwargs)

        self.__dict__.update(kwargs) # add attributes like: merlin_patch_size, merlin_image_size, merlin_num_channels

        self.max_processors = 3

        # Initialize the original subprocessor (Qwen2VLImageProcessor)
        self.original_processor = original_processor
        self.merge_size = self.original_processor.merge_size
        # self.image_processor = self.original_processor.image_processor # For passing the image_processor check in llamafactory/data/mm_plugin.py.

        # Initialize the first new-subprocessor
        self.new_processor_1 = new_processor_1
        self.new_processor_1_type = new_processor_1_type

        # Initialize the second new-subprocessor
        self.new_processor_2 = new_processor_2
        self.new_processor_2_type = new_processor_2_type

        # Initialize the third new-subprocessor
        self.new_processor_3 = new_processor_3
        self.new_processor_3_type = new_processor_3_type

    def preprocess(self, images: PIL.Image.Image, return_tensors: str = "pt", **kwargs):
        """
        Processes an image using two different underlying processors.

        Args:
            images (`ImageInput`): An image or list of images to process.
            return_tensors (`str`, *optional*): The type of tensors to return.

        Returns:
            BatchFeature: A dictionary-like object containing the outputs from both processors.
        """
        
        # 1. Process the image(s) with the first preprocessor
        # The `preprocess` methods already handle lists, so we don't need to loop.
        original_kwargs = {i:j for i,j in kwargs.items() if "volume" not in i}
        original_outputs = self.original_processor.preprocess(images, return_tensors=return_tensors, **original_kwargs)
        # original_outputs = self.original_processor.preprocess(images, return_tensors=return_tensors, **kwargs)

        # 2. Combine the results into a single dictionary
        combined_data = {}
        combined_data["pixel_values"] = original_outputs["pixel_values"]
        combined_data["image_grid_thw"] = original_outputs["image_grid_thw"]

        
        for i in range(self.max_processors):
            processor_index = i + 1

            # Use getattr() to retrieve the processor and its type
            processor = getattr(self, f"new_processor_{processor_index}", None) # Use default=None to avoid errors if attr doesn't exist
            processor_type = getattr(self, f"new_processor_{processor_index}_type", None)
            if processor is None and processor_type is None:
                continue

            if "dinov2" in processor_type or "medsiglip" in processor_type:
                # 3. Process the same original image(s) with the second preprocessor
                if isinstance(images[0], list): # This is for batched inference only in MedEvalKit-V2.
                    new_outputs_images = []
                    for one_images_list in images:
                        one_new_outputs = processor.preprocess(one_images_list, return_tensors=return_tensors)
                        new_outputs_images.append(one_new_outputs["pixel_values"])
                    new_outputs_images = torch.cat(new_outputs_images, dim = 0)
                    new_outputs_1 = BatchFeature(data={"pixel_values": new_outputs_images}, tensor_type=return_tensors)
                else:
                    new_outputs_1 = processor.preprocess(images, return_tensors=return_tensors)
                
                combined_data[f"pixel_values_{processor_type}"] = new_outputs_1["pixel_values"]

            elif "convnext" in processor_type:
                # 3. Process the same original image(s) with the second preprocessor
                if isinstance(images[0], PIL.Image.Image):
                    new_outputs_1 = torch.stack([processor(i) for i in images])
                    combined_data[f"pixel_values_{processor_type}"] = new_outputs_1

                elif isinstance(images[0], list): # This is for batched inference only in MedEvalKit-V2.
                    new_outputs_images = []
                    for one_images_list in images:
                        one_new_outputs = torch.stack([processor(i) for i in one_images_list])
                        new_outputs_images.append(one_new_outputs)
                    new_outputs_images = torch.cat(new_outputs_images, dim = 0)
                    combined_data[f"pixel_values_{processor_type}"] = new_outputs_images
            
            elif "merlin" in processor_type or "pe_3d" in processor_type:
                # assert "volume" in kwargs # We have volume so that merlin can process.
                
                if "volume" in kwargs:
                    # Add the batch dimension. After unsqueeze: [bsz, num_volumes, depths, 512, 512]
                    if isinstance(kwargs['volume'], list):
                        # combined_data[f"pixel_values_{processor_type}"] = torch.stack(kwargs['volume'], dim = 0)
                        if isinstance(kwargs['volume'][0], list):
                            vols = []
                            for one_vol in kwargs['volume']:
                                vols.append(torch.stack(one_vol))  
                            combined_data[f"pixel_values_{processor_type}"] = torch.cat(vols, dim = 0)   
                        else:
                            combined_data[f"pixel_values_{processor_type}"] = torch.stack(kwargs['volume'], dim = 0)
                    else:
                        combined_data[f"pixel_values_{processor_type}"] = \
                            kwargs["volume"].unsqueeze(0) if len(kwargs['volume'].shape) == 4 else kwargs["volume"] 

            else:
                assert False, f"{processor_type} processor type is still not supported."
            
            ### Test save_pretrained
            # self.save_pretrained("output/multi_enc_qwen2.5-vl-stage1-convnext_v4/tmp_test_save")
        
        return BatchFeature(data=combined_data, tensor_type=return_tensors)

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Saves the processor's configuration and its sub-processors.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
            
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. Create the main configuration dictionary
        # We store information on how to reload the sub-processors, not the objects themselves.
        config = {
            "processor_class": self.__class__.__name__,
            # "processor_class": self.processor_class,
            "new_processor_1_type": self.new_processor_1_type,
            "new_processor_2_type": self.new_processor_2_type,
            "new_processor_3_type": self.new_processor_3_type,
            # Store the class names for flexible loading
            "original_processor_class": self.original_processor.__class__.__name__,
            "new_processor_1_class": self.new_processor_1.__class__.__name__,
            "new_processor_2_class": self.new_processor_2.__class__.__name__ if self.new_processor_2 is not None else None,
            "new_processor_3_class": self.new_processor_3.__class__.__name__ if self.new_processor_3 is not None else None,
        }
        
            
        # 2. Save each sub-processor in its own directory
        original_processor_path = os.path.join(save_directory, "original_processor")
        self.original_processor.save_pretrained(original_processor_path, **kwargs)

        for i in range(self.max_processors):
            processor_index = i + 1

            # Use getattr() to retrieve the processor and its type
            processor = getattr(self, f"new_processor_{processor_index}", None) # Use default=None to avoid errors if attr doesn't exist
            processor_type = getattr(self, f"new_processor_{processor_index}_type", None)
            if processor is None and processor_type is None:
                continue
            
            processor_save_path = os.path.join(save_directory, f"new_processor_{processor_index}")

            if "dinov2" in processor_type or "medsiglip" in processor_type:
                processor.save_pretrained(processor_save_path, **kwargs)
            elif "convnext" in processor_type:
                self.save_transforms(
                    transforms_to_save = processor, 
                    save_path = processor_save_path, 
                    save_json_name = "preprocessor_config.json",
                )
            elif "merlin" in processor_type or "pe_3d" in processor_type:
                for key, value in self.__dict__.items():
                    if key.startswith("merlin_"):
                        # If an attribute's name starts with 'merlin_', add it to the config.
                        config[key] = value
                # pass # Merlin does not have a processor.
            else:
                assert False, f"{processor_type} processor saving not supported yet."
        

        # 3. Save the main processor's config file
        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(config, indent=2))

        print(f"ImageMultiProcessor saved to {save_directory}")


    def save_transforms(self, transforms_to_save, save_path, save_json_name):
        # --- Initialize an empty dictionary to hold our extracted config ---
        config_to_save = {}

        # --- Iterate through the list of transforms to find the ones we need ---
        # The Compose object holds its transforms in a `.transforms` list
        for transform_step in transforms_to_save.transforms:
            if isinstance(transform_step, T.Resize):
                config_to_save['resize_size'] = transform_step.size
                # Save the interpolation mode as a string
                config_to_save['interpolation'] = transform_step.interpolation.name.lower()
            
            elif isinstance(transform_step, T.CenterCrop):
                # The crop size is the final image size the model expects.
                # It's often a tuple (H, W), but open_clip usually wants a single int.
                size = transform_step.size
                config_to_save['image_size'] = size[0] if isinstance(size, tuple) else size
                
            elif isinstance(transform_step, T.Normalize):
                # Ensure mean and std are saved as plain Python lists, not tensors
                config_to_save['mean'] = list(transform_step.mean)
                config_to_save['std'] = list(transform_step.std)

        # --- Sanity check that we found everything ---
        required_keys = ['image_size', 'mean', 'std', 'interpolation']
        if not all(key in config_to_save for key in required_keys):
            raise RuntimeError("Could not automatically extract all required parameters from the transform pipeline.")

        # --- Now, save this extracted config to a JSON file ---
        os.makedirs(save_path, exist_ok=True)
        config_filepath = os.path.join(save_path, save_json_name)

        with open(config_filepath, 'w') as f:
            json.dump(config_to_save, f, indent=2)

        print(f"Processor configuration saved to: {config_filepath}")
        print("Saved config:", json.dumps(config_to_save, indent=2))