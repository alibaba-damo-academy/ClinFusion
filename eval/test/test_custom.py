import importlib.util
import sys
import os
from typing import List, Dict, Any
from pathlib import Path
import nibabel as nib # <-- ADDED IMPORT

# ==============================================================================
# 1. CustomWrapper CLASS DEFINITION (UNCHANGED)
# ==============================================================================

class CustomWrapper:
    """
    A generic wrapper for custom models. It now correctly handles packages with
    relative imports by adjusting sys.path and creating an __init__.py if needed.
    """
    def __init__(self, model_path: str, model_config: dict, generation_config: dict):
        model_def_path_str = model_config["model_definition_path"]
        class_name = model_config["model_class_name"]

        print(f"[CustomWrapper] Loading custom model adapter...")
        print(f"  -> Definition path: {model_def_path_str}")
        print(f"  -> Class name: {class_name}")

        model_def_path = Path(model_def_path_str).resolve()
        
        if not model_def_path.is_file():
            raise FileNotFoundError(f"Custom model definition file not found at: {model_def_path}")

        package_dir = model_def_path.parent
        package_name = package_dir.name
        module_name = model_def_path.stem

        init_py_path = package_dir / "__init__.py"
        if not init_py_path.is_file():
            print(f"WARNING: Package marker '{init_py_path}' not found.")
            print("         Creating it automatically to enable relative imports.")
            init_py_path.touch()

        package_root = package_dir.parent
        if str(package_root) not in sys.path:
            print(f"[CustomWrapper] Adding package root '{package_root}' to sys.path.")
            sys.path.insert(0, str(package_root))

        full_module_name = f"{package_name}.{module_name}"
        print(f"[CustomWrapper] Importing module as '{full_module_name}' to resolve package context.")
        custom_module = importlib.import_module(full_module_name)
        
        if not hasattr(custom_module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in module '{full_module_name}'")

        ModelAdapterClass = getattr(custom_module, class_name)

        self.model_adapter = ModelAdapterClass(
            model_path=model_path,
            model_config=model_config,
            generation_config=generation_config
        )
        print("[CustomWrapper] Custom model adapter initialized successfully.")

    def generate(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        print("\n[CustomWrapper] Delegating generation task to the loaded model adapter.")
        return self.model_adapter.generate(batch_data)

# ==============================================================================
# 2. MAIN TEST LOGIC (UPDATED)
# ==============================================================================

def main():
    """
    Main function to run the live test.
    """
    # --- Step 1: Define all required hardcoded paths ---
    MODEL_CHECKPOINT_PATH = "/mnt/workspace/workgroup/hangjie.yhj/code_medical/Lingshu-2/output/multi_enc_qwen2.5-vl-stage1-merlin_v4_8gpus_Merlin_h20/checkpoint-388"
    ADAPTER_FILE_PATH = "/mnt/workspace/workgroup/hangjie.yhj/code_medical/MedEvalKit-V2-25-11-07/models/Multi_Enc_Qwen2_5_VL_25_11_24/medevalkit_adapter.py"
    TEST_NIFTI_PATH = "/mnt/eff_nas/tangzhiwei.tzw/amos_0001.nii.gz"

    print("--- 1. Validating file paths ---")
    paths_to_check = {
        "Model checkpoint directory": (MODEL_CHECKPOINT_PATH, 'dir'),
        "Adapter definition file": (ADAPTER_FILE_PATH, 'file'),
        "Test NIfTI file": (TEST_NIFTI_PATH, 'file'),
    }
    all_paths_ok = True
    for name, (path_str, path_type) in paths_to_check.items():
        p = Path(path_str)
        if (path_type == 'dir' and p.is_dir()) or (path_type == 'file' and p.is_file()):
            print(f"✅ Found: {name}")
        else:
            print(f"❌ Missing: {name}. Expected at {path_str}")
            all_paths_ok = False

    if not all_paths_ok:
        print("\nError: One or more required files/directories are missing. Please check the paths and try again.")
        sys.exit(1)
    print("All paths validated successfully.\n")

    # --- Step 2: Set up configurations ---
    print("--- 2. Setting up test configurations ---")
    model_config = {
        "model_definition_path": ADAPTER_FILE_PATH,
        "model_class_name": "MedEvalKitAdapter"
    }
    generation_config = {
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512
    }

    # --- Step 3: Pre-load data and prepare sample batch ---
    # --- UPDATED: Pre-load the NIfTI data before creating the batch ---
    print("--- 3. Pre-loading NIfTI data into memory ---")
    preloaded_nifti_object = nib.load(TEST_NIFTI_PATH)
    print(f"Data from '{TEST_NIFTI_PATH}' loaded into a nibabel object.\n")
    
    batch_data = [
        {
            "messages": {
                "prompt": "Please generate a medical report for the following 3D scan",
                "nifti": [preloaded_nifti_object] # <-- Pass the loaded object
            }
        },
        {
            "messages": {
                "prompt": "What did you see in the provided content?",
                "nifti": [preloaded_nifti_object] # <-- Pass the file path
            }
        }
    ]
    print("Configurations and sample data (with both object and path) are ready.\n")

    # --- Step 4: Initialize the CustomWrapper and run generation ---
    print("--- 4. Initializing CustomWrapper (this will load the full model, may take time) ---")
    wrapper = CustomWrapper(
        model_path=MODEL_CHECKPOINT_PATH,
        model_config=model_config,
        generation_config=generation_config
    )
    
    results = wrapper.generate(batch_data)
    
    # --- Step 6: Print the final results ---
    print("\n" + "="*80)
    print("              LIVE MODEL GENERATION RESULTS")
    print("="*80)
    for i, result_text in enumerate(results):
        data_type = "Pre-loaded NIfTI Object" if i == 0 else "NIfTI File Path"
        print(f"\n--- Result for Sample {i+1} ({data_type} input) ---")
        print(result_text)
    print("\n" + "="*80)
    print("\n✅ Live test completed successfully.")

if __name__ == "__main__":
    main()
