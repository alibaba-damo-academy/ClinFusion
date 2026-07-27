import importlib.util
import sys
from typing import List, Dict, Any
from pathlib import Path

class CustomWrapper:
    """
    A generic wrapper for custom models. It correctly handles packages with
    relative imports by adjusting sys.path and creating an __init__.py if needed.
    """
    def __init__(self, model_path: str, model_config: dict, generation_config: dict):
        model_def_path_str = model_config["model_definition_path"]
        class_name = "MedEvalKitAdapter"

        print(f"[CustomWrapper] Loading custom model adapter...")
        print(f"  -> Definition path: {model_def_path_str}")
        print(f"  -> Class name: {class_name}")

        model_def_path = Path(model_def_path_str).resolve()
        
        if not model_def_path.is_file():
            raise FileNotFoundError(f"Custom model definition file not found at: {model_def_path}")

        # --- THIS IS THE CRUCIAL FIX FOR RELATIVE IMPORTS ---

        # 1. Identify the package directory and name
        package_dir = model_def_path.parent
        package_name = package_dir.name
        module_name = model_def_path.stem

        # 2. Ensure the directory is a package (important for Python to resolve '.')
        init_py_path = package_dir / "__init__.py"
        if not init_py_path.is_file():
            print(f"WARNING: Package marker '{init_py_path}' not found. Creating it to enable relative imports.")
            init_py_path.touch()

        # 3. Add the package's ROOT directory to sys.path so Python can find it.
        package_root = package_dir.parent
        if str(package_root) not in sys.path:
            print(f"[CustomWrapper] Adding package root '{package_root}' to sys.path.")
            sys.path.insert(0, str(package_root))

        # 4. Import the module using its full package path (e.g., 'MyModelPackage.adapter_file')
        full_module_name = f"{package_name}.{module_name}"
        print(f"[CustomWrapper] Importing module as '{full_module_name}' to resolve package context.")
        custom_module = importlib.import_module(full_module_name)
        
        # --- END OF FIX ---

        if not hasattr(custom_module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in module '{full_module_name}'")

        ModelAdapterClass = getattr(custom_module, class_name)

        self.model_adapter = ModelAdapterClass(
            model_path=model_path,
            model_config=model_config,
            generation_config=generation_config
        )

        self.min_image_size = model_config.get('min_image_size', 32)
        self.max_image_size = model_config.get('max_image_size', 1024)
        self.max_image_num = model_config.get('max_image_num', 10)

        print("[CustomWrapper] Custom model adapter initialized successfully.")

    def generate(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """
        Delegates the generation task to the loaded custom model adapter.
        """
        print("\n[CustomWrapper] Delegating generation task to the loaded model adapter.")
        return self.model_adapter.generate(batch_data)
