import torch.nn as nn
import torch
from transformers import AutoModel, PreTrainedModel, PretrainedConfig, AutoConfig
from typing import Tuple, Optional
from torchvision.transforms.functional import resize as tensor_resize
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from typing import Optional, List, Union, Tuple, Dict, Any
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from .multi_enc_utils import parse_local_field, parse_use_ffn, parse_expansion_factor, parse_cascade_mode
import numpy as np



# 1. Define a Custom Configuration Class
class VisionWrapperConfig(PretrainedConfig):
    """
    Configuration class for VisionWrapper. It stores the configuration
    of the underlying vision model and the target LLM hidden size.
    """
    # This model_type MUST match the string we use to register it.
    model_type = "multi_enc_vision_wrapper"

    def __init__(
        self,
        vision_model_config: dict = None,
        llm_hidden_size: int = None,
        tokens_per_second: int = None,
        spatial_merge_size: int = None,
        fusion_type: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_model_config = vision_model_config
        self.llm_hidden_size = llm_hidden_size
        self.tokens_per_second = tokens_per_second
        self.spatial_merge_size = spatial_merge_size
        self.fusion_type = fusion_type



### V5: support multi-enc processing (ConvNeXT + DINOv2) with hierarchical fusion
# 1. add multiple FFN definitions in init.
# 2. add logic for hierarchical fusion.

class VisionWrapper(nn.Module):
    """
    A wrapper module to make multiple vision encoders (like DINOv2) API-compatible.
    """
    # This attribute is needed for Hugging Face to identify it as a custom module
    _auto_class = "AutoModel"
    
    # This tells from_pretrained which config class to use
    config_class = VisionWrapperConfig

    def __init__(
        self, 
        config: VisionWrapperConfig, 
        original_enc, 
        dinov2 = None,
        convnext = None,
        pe_3d = None,
    ):
        super().__init__()
        self.config = config
        self.original_enc = original_enc
        self.original_enc_hid_size = self.config.llm_hidden_size # self.config.vision_model_config["original_enc"]
        self.dinov2    = dinov2
        self.convnext  = convnext
        self.pe_3d     = pe_3d
        self.fusion_type = config.fusion_type

        if dinov2 is not None:
            self.dinov2_latent_shape = (16, 16)
            self.dinov2_hid_size = self.config.vision_model_config["dinov2"]["hidden_size"]
        
        if convnext is not None:
            force_image_size = self.config.vision_model_config["convnext"]["model_cfg"]["vision_cfg"]["image_size"]
            self.convnext_latent_shape = (force_image_size//4, force_image_size//4) # (768//4, 768//4)
            # self.convnext_hid_size = self.config.vision_model_config["convnext"]["hidden_sizes"][-1]
            self.convnext_hid_size = sum(self.config.vision_model_config["convnext"]["model_cfg"]["hidden_sizes"])
            self.convnext_expansion_factor = parse_expansion_factor(self.fusion_type)
        
        
        if pe_3d is not None:
            # import ipdb; ipdb.set_trace()
            self.pe_3d_hid_size = self.config.vision_model_config["pe_3d"]['width']
        
        if dinov2 is not None and convnext is not None:
            # When multiple encoders are enables, we only support cross-attn.
            assert "cross_attn" in self.fusion_type, f"{self.fusion_type} is not supported when using multiple encoders."
        
        ### number of aux encoders (need to be modified if new encoders are introduced.)
        self.num_aux_encoders = sum([1 if dinov2 is not None else 0, 
                                     1 if convnext is not None else 0, 
                                     1 if pe_3d is not None else 0])
        
        self.dtype = self.original_enc.dtype
        self.device = self.original_enc.device

        ### _init_fusion_module
        if self.fusion_type == "channel_concat":
            enc_hidden_sum = self.config.llm_hidden_size
            for cfg_name, cfg in self.config.vision_model_config.items():
                if cfg_name != "original_enc":
                    enc_hidden_sum += cfg["hidden_size"]

            self.fusion_projector = nn.Linear(
                enc_hidden_sum,
                self.config.llm_hidden_size
            )
        elif "cross_attn" in self.fusion_type:
        # elif self.fusion_type == "cross_attn":
            self.use_ffn = parse_use_ffn(self.fusion_type) # = False # for debugging
            self.local_field = parse_local_field(self.fusion_type)

            if dinov2 is not None:
                self.dinov2_pre_projector = nn.Linear(self.dinov2_hid_size, self.original_enc_hid_size)
                # Qwen features will query DINOv2 features
                self.dinov2_cross_attn = nn.MultiheadAttention(
                    embed_dim=self.original_enc_hid_size, # Dimension of Q, K, V
                    num_heads=16, # Example, should be configurable
                    batch_first=True,
                    dropout=0.1
                )
            
            if convnext is not None:
                self.convnext_pre_projector = nn.Linear(self.convnext_hid_size, self.original_enc_hid_size)
                # Qwen features will query convnext features
                self.convnext_cross_attn = nn.MultiheadAttention(
                    embed_dim=self.original_enc_hid_size, # Dimension of Q, K, V
                    num_heads=16, # Example, should be configurable
                    batch_first=True,
                    dropout=0.1
                )
            
            if pe_3d is not None:
                self.pe_3d_pre_projector = nn.Linear(self.pe_3d_hid_size, self.original_enc_hid_size)
                # Qwen features will query pe_3d features
                self.pe_3d_cross_attn = nn.MultiheadAttention(
                    embed_dim=self.original_enc_hid_size, # Dimension of Q, K, V
                    num_heads=16, # Example, should be configurable
                    batch_first=True,
                    dropout=0.1
                )

            if "cascade" in self.fusion_type:
                assert self.num_aux_encoders >= 2
                self.cascade_mode =  parse_cascade_mode(self.fusion_type) # 1, 2

                if self.use_ffn:
                    ffn_list = [
                        nn.Sequential(
                            nn.LayerNorm(self.original_enc_hid_size),
                            nn.Linear(self.original_enc_hid_size, self.original_enc_hid_size * 4),
                            nn.GELU(),
                            nn.Linear(self.original_enc_hid_size * 4, self.original_enc_hid_size)
                        ) for _ in range(self.num_aux_encoders)
                    ]
                else:
                    ffn_list = [nn.Identity() for _ in range(self.num_aux_encoders)]
                self.fusion_ffn = nn.ModuleList(ffn_list)

            else:
                self.fusion_ffn = nn.Sequential(
                    nn.LayerNorm(self.original_enc_hid_size),
                    nn.Linear(self.original_enc_hid_size, self.original_enc_hid_size * 4),
                    nn.GELU(),
                    nn.Linear(self.original_enc_hid_size * 4, self.original_enc_hid_size)
                ) if self.use_ffn else nn.Identity()

            # Final projector to LLM dimension
            self.fusion_projector = nn.Linear(self.original_enc_hid_size, self.config.llm_hidden_size)
            
        elif self.fusion_type == "move_ffn":
            if dinov2 is not None:
                self.dinov2_pre_projector = nn.Linear(self.dinov2_hid_size, self.original_enc_hid_size)
            
            self.num_vision_experts = 1
            self.num_vision_experts = self.num_vision_experts + 1 if self.dinov2 is not None else self.num_vision_experts
            self.moe_router = nn.Linear(self.original_enc_hid_size, self.num_vision_experts)

            self.fusion_ffn = nn.Sequential(
                nn.LayerNorm(self.original_enc_hid_size),
                nn.Linear(self.original_enc_hid_size, self.original_enc_hid_size * 4),
                nn.GELU(),
                nn.Linear(self.original_enc_hid_size * 4, self.original_enc_hid_size)
            )
            # Final projector has same input size as experts
            self.fusion_projector = nn.Linear(self.original_enc_hid_size, self.config.llm_hidden_size)
        else:
            assert False, "Fusion type is not implemented."




def register_multi_enc_vision_wrapper():
    """Registers the custom classes with the Auto* classes."""
    AutoConfig.register(VisionWrapperConfig.model_type, VisionWrapperConfig)
    AutoModel.register(VisionWrapperConfig, VisionWrapper)
    print(f"Successfully registered {VisionWrapperConfig.model_type} with AutoConfig and AutoModel.")

register_multi_enc_vision_wrapper()
