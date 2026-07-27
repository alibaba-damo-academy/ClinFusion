from transformers import (
    AutoConfig,
    AutoModel
)
from .modeling_multi_enc_qwen2_5_vl import VisionWrapperConfig, VisionWrapper
from .openclip_encoder import OpenCLIPVisionTower

from .medical_vit import create_ssl_large_p8_3d_pe_model

from packaging.version import parse as parse_version
import transformers
if parse_version(transformers.__version__) >= parse_version("4.56.0"): # dev version cannot be recognized.
    from .modeling_multi_enc_qwen3_vl import Qwen3VL_VisionWrapper


# V2: Support multiple aux encoders init.
def add_new_enc_to_model(
    model,
    model_args,
):
    new_enc_type = [t for t in [model_args.new_encoder_type, model_args.new_encoder_2_type, model_args.new_encoder_3_type] if t is not None]
    new_enc_path = [t for t in [model_args.new_encoder_path, model_args.new_encoder_2_path, model_args.new_encoder_3_path] if t is not None]
    assert len(new_enc_type) == len(new_enc_path)

    vision_model_config = {"original_enc": model.config.vision_config.to_dict()}

    enc_kwargs = {"original_enc": model.visual}

    for enc_type, enc_path in zip(new_enc_type, new_enc_path):
        if "dinov2" in enc_type:
            ### Config 
            new_enc_config = AutoConfig.from_pretrained(enc_path, trust_remote_code=True)
            vision_model_config[enc_type] = new_enc_config.to_dict()

            ### Model
            new_enc_model = AutoModel.from_pretrained(
                enc_path,
                torch_dtype = model_args.compute_dtype,
                trust_remote_code = True
            ).to(model.device)
            enc_kwargs[enc_type] = new_enc_model

        elif "convnext" in enc_type:
            force_image_size = model_args.convnext_image_size
            ### Config
            openclip_convnext = OpenCLIPVisionTower(
                vision_tower = "convnext-large-d-320", # "convnext_xxlarge", 
                model_path = enc_path,
                force_image_size = force_image_size,
            ).to(model.device)
            vision_model_config[enc_type] = openclip_convnext.config

            ### Model
            enc_kwargs[enc_type] = openclip_convnext
        
        elif "pe_3d" in enc_type:
            new_enc_model, new_enc_config = create_ssl_large_p8_3d_pe_model(
                enc_path, 
                patch_size = model_args.merlin_patch_size,
                return_config = True
            )
            new_enc_model = new_enc_model.to(model.device)

            vision_model_config[enc_type] = new_enc_config
            enc_kwargs[enc_type] = new_enc_model

        else:
            assert False, f"Model type {enc_type} not supported yet."

    ### Set the config
    # llm_hidden_size = model.config.hidden_size
    llm_hidden_size = getattr(model.config, "hidden_size", None)
    if llm_hidden_size is None: # defend against Qwen3-VL
        llm_hidden_size = getattr(model.config.text_config, "hidden_size")
    
    tokens_per_second = None 
    if "Qwen2_5_VL" in model.config.architectures[0]:
        tokens_per_second = model.config.vision_config.tokens_per_second

    wrapper_config = VisionWrapperConfig(
        vision_model_config=vision_model_config,
        llm_hidden_size=llm_hidden_size,
        tokens_per_second=tokens_per_second,
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
        fusion_type=model_args.multi_enc_fusion_type,
    )
    # config.vision_config = wrapper_config
    model.config.vision_config = wrapper_config

    ### Set the visual encoder
    VisionWrapperClass = None
    if "Qwen2_5_VL" in model.config.architectures[0]:
        VisionWrapperClass = VisionWrapper
    elif "Qwen3VL" in model.config.architectures[0]:
        VisionWrapperClass = Qwen3VL_VisionWrapper
    else:
        assert f"There is no VisionWrapper for {model.config.architectures[0]}."

    vision_wrapper = VisionWrapperClass(
        config=wrapper_config,
        **enc_kwargs,
    ).to(device=model.device, dtype=model.dtype)
    model.set_visual(vision_wrapper)

    return model