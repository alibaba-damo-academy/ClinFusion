from .qwenvl import QwenVLWrapper
from .api import APIWrapper
from .hulumed import HuluMedWrapper
from .custom import CustomWrapper

# This dictionary maps a model_type string from your config to the correct wrapper class.
MODEL_WRAPPERS = {
    "qwenvl": QwenVLWrapper,
    "api": APIWrapper,
    "hulumed": HuluMedWrapper,
    "custom": CustomWrapper
}

def get_wrapper(model_type: str, model_path: str, model_config: dict, generation_config: dict):
    """
    Factory function to get an instance of a model wrapper based on the model_type.

    Args:
        model_type (str): The type of the model (e.g., 'qwenvl').
        model_path (str): Path to the model checkpoint.
        model_config (dict): Configuration specific to the model and engine.
        generation_config (dict): Configuration for the text generation process.

    Returns:
        An instance of the appropriate model wrapper class.
    """
    if model_type not in MODEL_WRAPPERS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available types are: {list(MODEL_WRAPPERS.keys())}"
        )

    if model_type == "custom":

        if "model_definition_path" not in model_config:
            raise ValueError("model_type 'custom' requires 'model_definition_path' to be set in model_config.")
    
    # Get the correct class from the dictionary
    WrapperClass = MODEL_WRAPPERS[model_type]
    
    # Instantiate and return the wrapper
    return WrapperClass(
        model_path=model_path,
        model_config=model_config,
        generation_config=generation_config
    )