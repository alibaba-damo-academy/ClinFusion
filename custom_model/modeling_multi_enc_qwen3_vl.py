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

from .modeling_multi_enc_qwen2_5_vl import VisionWrapper, VisionWrapperConfig

# --- Define a boolean flag for easy checking elsewhere in the code ---
from packaging.version import parse as parse_version
import transformers
if parse_version(transformers.__version__) >= parse_version("4.56.0"): # dev version cannot be recognized.
    # If the version is new enough, import the Qwen3-VL classes
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast
    from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.utils.generic import check_model_inputs
    from transformers.cache_utils import Cache
    from transformers.processing_utils import Unpack
    print("Transformers version > 4.57.0 detected. Importing Qwen3-VL modules.")


# -------------------------------------------------------------------------------------------------------- #
# ----------------------------------- Multi-enc support for Qwen3-VL ------------------------------------- #
# -------------------------------------------------------------------------------------------------------- #

class Qwen3VL_VisionWrapper(VisionWrapper):
    def __init__(
        self,
        config: VisionWrapperConfig, 
        original_enc, 
        dinov2 = None,
        convnext = None,
        pe_3d = None,
    ):
        super().__init__(config, original_enc, dinov2, convnext, pe_3d)
        
        ### Define some additional params for DeepStack Fusion.
        self.deepstack_layers = 3

        if "cross_attn" in self.fusion_type:

            if dinov2 is not None:
                # Qwen features will query DINOv2 features
                self.dinov2_cross_attn_deepstack = nn.ModuleList([nn.MultiheadAttention(
                    embed_dim=self.original_enc_hid_size, # Dimension of Q, K, V
                    num_heads=16, # Example, should be configurable
                    batch_first=True,
                    dropout=0.1
                ) for _ in range(self.deepstack_layers)])
            
            if convnext is not None:
                # Qwen features will query convnext features
                self.convnext_cross_attn_deepstack = nn.ModuleList([nn.MultiheadAttention(
                    embed_dim=self.original_enc_hid_size, # Dimension of Q, K, V
                    num_heads=16, # Example, should be configurable
                    batch_first=True,
                    dropout=0.1
                ) for _ in range(self.deepstack_layers)])
            
            
            if pe_3d is not None:
                # Qwen features will query pe_3d features
                self.pe_3d_cross_attn_deepstack = nn.ModuleList([nn.MultiheadAttention(
                    embed_dim=self.original_enc_hid_size, # Dimension of Q, K, V
                    num_heads=16, # Example, should be configurable
                    batch_first=True,
                    dropout=0.1
                ) for _ in range(self.deepstack_layers)])


            if "cascade" in self.fusion_type:
                assert self.num_aux_encoders >= 2
                self.cascade_mode = parse_cascade_mode(self.fusion_type) # 1, 2

                if self.use_ffn:
                    self.fusion_ffn_deepstack = nn.ModuleList([
                            nn.ModuleList([
                                nn.Sequential(
                                    nn.LayerNorm(self.original_enc_hid_size),
                                    nn.Linear(self.original_enc_hid_size, self.original_enc_hid_size * 4),
                                    nn.GELU(),
                                    nn.Linear(self.original_enc_hid_size * 4, self.original_enc_hid_size)
                                ) for _ in range(self.num_aux_encoders)]) for _ in range(self.deepstack_layers)
                        ]
                    )
                else:
                    self.fusion_ffn_deepstack = nn.ModuleList([
                            nn.ModuleList([nn.Identity() for _ in range(self.num_aux_encoders)]) for _ in range(self.deepstack_layers)
                        ]
                    )
            
            # Final projector to LLM dimension
            self.deepstack_fusion_projector = nn.ModuleList([nn.Linear(self.original_enc_hid_size, self.config.llm_hidden_size) for _ in range(self.deepstack_layers)])

        else:
            assert False, "Fusion type is not implemented."

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        grid_thw: torch.Tensor, 
        pixel_values_dinov2: torch.Tensor = None,
        pixel_values_convnext: torch.Tensor = None,
        pixel_values_pe_3d: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:

        # original_image_embeds = self.original_enc(pixel_values, grid_thw, **kwargs) # [846, 3584]
        original_image_embeds, deepstack_image_embeds = self.original_enc(pixel_values, grid_thw=grid_thw)
        # example:
        # image_embeds shape: [5379, 4096] 
        # deepstack_image_embeds shape: [[5379, 4096], [5379, 4096], [5379, 4096]]
        # grid_thw content: [[ 1, 58, 40], [ 1, 58, 40], [ 1, 90, 26], [ 1, 36, 64], [ 1, 42, 56],
        #                    [ 1, 40, 58], [ 1, 58, 40], [ 1, 40, 58], [ 1, 40, 58], [ 1, 20, 30]]

        ### Define processing flag and their order
        if "cascade" in self.fusion_type:
            ### If fusion is cascade, then processing_flags DEFINES the order.
            dinov2_processing    = True if (pixel_values_dinov2 is not None) and (self.dinov2 is not None) else False
            convnext_processing  = True if (pixel_values_convnext is not None) and (self.convnext is not None) else False
            pe_3d_processing    = True if (pixel_values_pe_3d is not None) and (self.pe_3d is not None) else False

            # Choose 2d encoder processing order
            processing_flags = {
                1: {"dinov2_processing": dinov2_processing,       "convnext_processing": convnext_processing},
                2: {"convnext_processing": convnext_processing,   "dinov2_processing": dinov2_processing},
            }[self.cascade_mode]
            
            # Add 3d encoder processing if it is used.
            if self.pe_3d is not None:
                processing_flags["pe_3d_processing"] = pe_3d_processing

            ### Init fusion embeds
            fused_embeddings_dict = {} # store fused embeddings for multi-enc fusion.
            fused_embeddings_dict["original_image_embeds"] = original_image_embeds
            fused_deepstack_embeddings_dict = {} # store fused embeddings for multi-enc fusion.
            fused_deepstack_embeddings_dict["deepstack_image_embeds"] = deepstack_image_embeds

        else:
            assert False, "Other types are not supported yet."


        ### Perform encoder inference and fusion.
        if "cascade" in self.fusion_type:
            # ----------------------------- Start of cascade inference and fusion ----------------------------- #
            assert "cross_attn_local" in self.fusion_type # currently, we only support this.
            processing_ordered_list = [k for k,v in processing_flags.items() if v is True]
            base_query_embeddings = original_image_embeds # This will serve as the query for the first processing
            deepstack_query_embeddings = deepstack_image_embeds
            
            for enc_idx, processing_enc in enumerate(processing_ordered_list):
                # print(enc_idx, processing_enc, self.cascade_mode)

                if processing_enc == "dinov2_processing":
                    dinov2_image_embeds = self.dinov2(pixel_values_dinov2, return_dict=True)['last_hidden_state'][:,1:] # [bsz, 256, 1024]
                    # odict_keys(['last_hidden_state', 'pooler_output'])
                    dinov2_image_embeds = dinov2_image_embeds.transpose(1, 2)
                    dinov2_image_embeds = dinov2_image_embeds.reshape(dinov2_image_embeds.shape[:2] + self.dinov2_latent_shape) # [2, 1024, 16, 16]
                    
                    resized_dinov2_image_embeds = []
                    for one_dinov2_embeds, one_thw in zip(dinov2_image_embeds, grid_thw):
                        original_enc_shape = (one_thw[-2] // 2, one_thw[-1] // 2)
                        resized_dinov2_image_embed = tensor_resize(one_dinov2_embeds, original_enc_shape) # original_image_embeds will reduce the spatial size by 2.
                        resized_dinov2_image_embed = resized_dinov2_image_embed.view(-1, original_enc_shape[0] * original_enc_shape[1])
                        resized_dinov2_image_embeds.append(resized_dinov2_image_embed)
                    resized_dinov2_image_embeds = torch.cat(resized_dinov2_image_embeds, dim = 1).transpose(0, 1)

                    if "cross_attn_local" in self.fusion_type:
                        # -------------------- Original image feature fusion -------------------- #
                        ### V2: One large mask without memory issue. (I still do not know why.)
                        aligned_resized_dinov2_image_embeds = self.dinov2_pre_projector(resized_dinov2_image_embeds)

                        batch_cumsum = np.cumsum([((one_thw[-2] // 2) * (one_thw[-1] // 2)).item() for one_thw in grid_thw])
                        batch_cumsum = np.insert(batch_cumsum, 0, 0, axis=0)

                        kv_list = []
                        for i in range(len(batch_cumsum) - 1):
                            i_aligned_resized_dinov2_image_embeds = aligned_resized_dinov2_image_embeds[batch_cumsum[i]:batch_cumsum[i+1]].contiguous()
                            h, w = grid_thw[i][-2] // 2, grid_thw[i][-1] // 2 # self.original_enc_hid_size

                            ### Locality features extraction.
                            local_dinov2_image_embeds = i_aligned_resized_dinov2_image_embeds\
                                .reshape(1, h, w, self.original_enc_hid_size)\
                                .permute(0, 3, 1, 2)
                            local_kv_neighborhoods = F.unfold(
                                local_dinov2_image_embeds, 
                                kernel_size=self.local_field, 
                                padding=self.local_field // 2, 
                                stride=1
                            ) 
                                # shape: [1, 3584 * self.local_field**2, h * w]
                            local_kv_neighborhoods = local_kv_neighborhoods.reshape(
                                self.original_enc_hid_size, self.local_field**2, h * w
                            ).permute(2, 1, 0)
                                # shape: [h * w, self.local_field**2, 3584]
                            kv_list.append(local_kv_neighborhoods)
                        kv = torch.cat(kv_list, dim = 0)
                            
                        ### Locality attention
                        query = base_query_embeddings.unsqueeze(1) # shape: [h * w, 1, 3584]
                        # query = original_image_embeds.unsqueeze(1) # shape: [h * w, 1, 3584]
                        attn_output, attn_weights_dinov2 = self.dinov2_cross_attn(
                            query = query, 
                            key = kv, 
                            value = kv
                        ) # shape: [h * w, 1, 3584]

                        base_query_embeddings = base_query_embeddings + attn_output.squeeze(1)
                        base_query_embeddings = base_query_embeddings + self.fusion_ffn[enc_idx](base_query_embeddings)

                        fused_embeddings_dict["fused_embeddings_dinov2"] = base_query_embeddings

                        # -------------------- DeepStack image feature fusion -------------------- #
                        # import ipdb; ipdb.set_trace()
                        if getattr(self, "dinov2_cross_attn_deepstack", None) is not None:
                            fused_deepstack_query_embeddings = []
                            for deepstack_idx, one_deepstack_query in enumerate(deepstack_query_embeddings):
                                one_deepstack_query = one_deepstack_query.unsqueeze(1) # shape: [h * w, 1, 3584]
                                attn_output, _ = self.dinov2_cross_attn_deepstack[deepstack_idx](
                                    query = one_deepstack_query, 
                                    key = kv, 
                                    value = kv
                                )

                                one_deepstack_query = one_deepstack_query.squeeze(1) + attn_output.squeeze(1)
                                one_deepstack_query = one_deepstack_query + \
                                                    self.fusion_ffn_deepstack[deepstack_idx][enc_idx](one_deepstack_query)
                                
                                fused_deepstack_query_embeddings.append(one_deepstack_query)
                            
                            deepstack_query_embeddings = fused_deepstack_query_embeddings
                            fused_deepstack_embeddings_dict[f"fused_embeddings_deepstack_{enc_idx}"] = deepstack_query_embeddings
                
                elif processing_enc == "convnext_processing":
                    ### Encode convnext feature
                    convnext_image_embeds = self.convnext(pixel_values_convnext) # [2, 5760, 192, 192]

                    original_expan_hw = [
                        (one_thw[-2] // 2 * self.convnext_expansion_factor, one_thw[-1] // 2 * self.convnext_expansion_factor) \
                        for one_thw in grid_thw
                    ]

                    resized_convnext_image_embeds = []
                    for one_convnext_embeds, convnext_hw in zip(convnext_image_embeds, original_expan_hw):
                        resized_convnext_image_embed = tensor_resize(one_convnext_embeds, convnext_hw) 
                        resized_convnext_image_embed = resized_convnext_image_embed.view(-1, convnext_hw[0] * convnext_hw[1])
                        resized_convnext_image_embeds.append(resized_convnext_image_embed)
                    resized_convnext_image_embeds = torch.cat(resized_convnext_image_embeds, dim = 1).transpose(0, 1) 
                    # shape: [sequence, self.convnext_hid_size]

                    if "cross_attn_local" in self.fusion_type:
                        # -------------------- Original image feature fusion -------------------- #

                        ### V2: One large mask without memory issue. (I still do not know why.)
                        aligned_resized_convnext_image_embeds = self.convnext_pre_projector(resized_convnext_image_embeds)

                        batch_cumsum = np.cumsum([(one_thw[-2] * one_thw[-1]).item() for one_thw in original_expan_hw])
                        batch_cumsum = np.insert(batch_cumsum, 0, 0, axis=0)

                        kv_list = []
                        for i in range(len(batch_cumsum) - 1):
                            i_aligned_resized_convnext_image_embeds = aligned_resized_convnext_image_embeds[batch_cumsum[i]:batch_cumsum[i+1]].contiguous()
                            h, w         = original_expan_hw[i][-2], original_expan_hw[i][-1]  # self.original_enc_hid_size
                            ori_h, ori_w = grid_thw[i][-2] // 2, grid_thw[i][-1] // 2 # self.original_enc_hid_size

                            ### Locality features extraction.
                            local_convnext_image_embeds = i_aligned_resized_convnext_image_embeds\
                                .reshape(1, h, w, self.original_enc_hid_size)\
                                .permute(0, 3, 1, 2)
                            local_kv_neighborhoods = F.unfold(
                                local_convnext_image_embeds, 
                                kernel_size = self.convnext_expansion_factor, 
                                padding = 0, 
                                stride = self.convnext_expansion_factor
                            ) 
                                # shape: [1, 3584 * self.convnext_expansion_factor**2, ori_h * ori_w]
                            # print("local_convnext_image_embeds ", local_convnext_image_embeds.shape)
                            local_kv_neighborhoods = local_kv_neighborhoods.reshape(
                                self.original_enc_hid_size, self.convnext_expansion_factor**2, (ori_h * ori_w)
                            ).permute(2, 1, 0)
                                # shape: [ori_h * ori_w, self.convnext_expansion_factor**2, 3584]
                            kv_list.append(local_kv_neighborhoods)
                        kv = torch.cat(kv_list, dim = 0)

                        ### Locality attention
                        query = base_query_embeddings.unsqueeze(1) 
                        attn_output, attn_weights_convnext = self.convnext_cross_attn(
                            query = query, # shape: [ori_h * ori_w, 1, 3584]
                            key = kv,      # shape: [ori_h * ori_w, self.convnext_expansion_factor**2, 3584]
                            value = kv,    # shape: [ori_h * ori_w, self.convnext_expansion_factor**2, 3584]
                        ) # shape: [ori_h * ori_w, 1, 3584]

                        base_query_embeddings = base_query_embeddings + attn_output.squeeze(1)
                        base_query_embeddings = base_query_embeddings + self.fusion_ffn[enc_idx](base_query_embeddings)

                        fused_embeddings_dict["fused_embeddings_convnext"] = base_query_embeddings


                        # -------------------- DeepStack image feature fusion -------------------- #
                        # import ipdb; ipdb.set_trace()
                        if getattr(self, "convnext_cross_attn_deepstack", None) is not None:
                            fused_deepstack_query_embeddings = []
                            for deepstack_idx, one_deepstack_query in enumerate(deepstack_query_embeddings):
                                one_deepstack_query = one_deepstack_query.unsqueeze(1) # shape: [h * w, 1, 3584]
                                attn_output, _ = self.convnext_cross_attn_deepstack[deepstack_idx](
                                    query = one_deepstack_query, 
                                    key = kv, 
                                    value = kv
                                )

                                one_deepstack_query = one_deepstack_query.squeeze(1) + attn_output.squeeze(1)
                                one_deepstack_query = one_deepstack_query + \
                                                    self.fusion_ffn_deepstack[deepstack_idx][enc_idx](one_deepstack_query)
                                
                                fused_deepstack_query_embeddings.append(one_deepstack_query)
                            
                            deepstack_query_embeddings = fused_deepstack_query_embeddings
                            fused_deepstack_embeddings_dict[f"fused_embeddings_deepstack_{enc_idx}"] = deepstack_query_embeddings

                elif processing_enc == "pe_3d_processing":
                    pe_3d_image_embeds = self.pe_3d.encode_image(pixel_values_pe_3d, retain_thw = True) # [4, 1, 32, 256, 256] -> [4, 1024, 4, 32, 32] 
                    pe_3d_bs, _, pe_3d_ft_depth, _, _ = pe_3d_image_embeds.shape
                    pe_3d_image_embeds = pe_3d_image_embeds.permute(0, 2, 1, 3, 4) # [4, 1024, 4, 32, 32] -> [4, 4, 1024, 32, 32] 

                    resized_pe_3d_image_embeds = []
                    original_enc_shape = (grid_thw[0][-2] // 2, grid_thw[0][-1] // 2)
                    for one_pe_3d_embeds in pe_3d_image_embeds:
                        resized_pe_3d_image_embed = tensor_resize(one_pe_3d_embeds, original_enc_shape) # original_image_embeds will reduce the spatial size by 2.
                        resized_pe_3d_image_embed = resized_pe_3d_image_embed.view(pe_3d_ft_depth, -1, original_enc_shape[0] * original_enc_shape[1])
                        resized_pe_3d_image_embeds.append(resized_pe_3d_image_embed)
                    resized_pe_3d_image_embeds = torch.stack(resized_pe_3d_image_embeds).transpose(2, 3) # [bs, 4, 8*8, 1024]

                    if "cross_attn_local" in self.fusion_type:
                        # -------------------- Original image feature fusion -------------------- #

                        ### V2: One large mask without memory issue. (I still do not know why.)
                        aligned_resized_pe_3d_image_embeds = self.pe_3d_pre_projector(resized_pe_3d_image_embeds)

                        kv_list = []
                        for i in range(pe_3d_bs):
                            ### Locality features extraction.
                            i_aligned_resized_pe_3d_image_embeds = aligned_resized_pe_3d_image_embeds[i]
                            local_pe_3d_image_embeds = i_aligned_resized_pe_3d_image_embeds\
                                .reshape(pe_3d_ft_depth, original_enc_shape[0], original_enc_shape[1], -1)\
                                .permute(0, 3, 1, 2)  # [4, 4096, 8, 8]

                            local_kv_neighborhoods = F.unfold(
                                local_pe_3d_image_embeds, 
                                kernel_size = self.local_field, 
                                padding = self.local_field // 2, 
                                stride = 1
                            ) # shape: [pe_3d_ft_depth, 4096 * self.local_field**2, h * w]
                            local_kv_neighborhoods = local_kv_neighborhoods.reshape(
                                pe_3d_ft_depth, self.original_enc_hid_size, self.local_field**2, -1
                            ).permute(3, 0, 2, 1)
                                # [h * w, pe_3d_ft_depth, self.local_field**2, 4096]
                            local_kv_neighborhoods = torch.flatten(local_kv_neighborhoods, start_dim=1, end_dim=2)
                                # [h * w, pe_3d_ft_depth * self.local_field**2, 4096]
                            kv_list.append(local_kv_neighborhoods)
                        

                        pe_3d_repeat_times = len(grid_thw) // pe_3d_bs # the number of sampled images in one 3D CT.
                        kv = []
                        for one_kv in kv_list:
                            for _ in range(pe_3d_repeat_times):
                                kv.append(one_kv.clone()) # Explicitly clone the tensor
                        kv = torch.cat(kv, dim = 0)
                        

                        ### Locality attention
                        # query = original_image_embeds.unsqueeze(1) # shape: [h * w, 1, 3584]
                        query = base_query_embeddings.unsqueeze(1)
                        attn_output, _ = self.pe_3d_cross_attn(
                            query = query, 
                            key = kv, 
                            value = kv
                        ) # shape: [h * w, 1, 3584]

                        base_query_embeddings = base_query_embeddings + attn_output.squeeze(1)
                        base_query_embeddings = base_query_embeddings + self.fusion_ffn[enc_idx](base_query_embeddings)

                        fused_embeddings_dict["fused_embeddings_pe_3d"] = base_query_embeddings


                        # -------------------- DeepStack image feature fusion -------------------- #
                        # import ipdb; ipdb.set_trace()
                        if getattr(self, "pe_3d_cross_attn_deepstack", None) is not None:
                            fused_deepstack_query_embeddings = []
                            for deepstack_idx, one_deepstack_query in enumerate(deepstack_query_embeddings):
                                one_deepstack_query = one_deepstack_query.unsqueeze(1) # shape: [h * w, 1, 3584]
                                attn_output, _ = self.pe_3d_cross_attn_deepstack[deepstack_idx](
                                    query = one_deepstack_query, 
                                    key = kv, 
                                    value = kv
                                )

                                one_deepstack_query = one_deepstack_query.squeeze(1) + attn_output.squeeze(1)
                                one_deepstack_query = one_deepstack_query + \
                                                    self.fusion_ffn_deepstack[deepstack_idx][enc_idx](one_deepstack_query)
                                
                                fused_deepstack_query_embeddings.append(one_deepstack_query)
                            
                            deepstack_query_embeddings = fused_deepstack_query_embeddings
                            fused_deepstack_embeddings_dict[f"fused_embeddings_deepstack_{enc_idx}"] = deepstack_query_embeddings
                    
                else:
                    assert False, f"{processing_enc} is not supported."

            ### Obtain the final fused embeddings for final LLM projection.
            fused_embeddings = list(fused_embeddings_dict.values())[-1]
            fused_deepstack_embeddings = list(fused_deepstack_embeddings_dict.values())[-1]
            
            # -----------------------------  End of cascade inference and fusion  -----------------------------#
        
        ### Final projection to original dimension
        fused_embeddings = self.fusion_projector(fused_embeddings)
        
        if getattr(self, "deepstack_fusion_projector", None) is not None:
            fused_deepstack_embeddings = [self.deepstack_fusion_projector[fde_idx](fde) for fde_idx, fde in enumerate(fused_deepstack_embeddings)]

        return fused_embeddings, fused_deepstack_embeddings


    

class MultiEnc_Qwen3VLModel(Qwen3VLModel):
    """
    一个继承自 Qwen3VLModel 的类，具有相同的结构，但重写了 forward 方法。
    它可以从一个现有的 Qwen3VLModel 实例无缝初始化。
    """

    @classmethod
    def from_qwen_model(cls, base_model: Qwen3VLModel) -> "MultiEnc_Qwen3VLModel":
        """
        Memory-efficient factory method that MODIFIES an existing Qwen3VLModel
        instance IN-PLACE to become a MultiEnc_Qwen3VLModel.

        Args:
            base_model (Qwen3VLModel): The existing model instance to be upgraded.
            model_args: Arguments needed to configure the new encoders.

        Returns:
            The SAME model object, but now with added layers and a new class.
        """
        # logger.info(f"Performing memory-efficient IN-PLACE upgrade of a {base_model.__class__.__name__} to {cls.__name__}.")

        # --- This is where the magic happens ---

        # Step 2: Change the class of the object at runtime.
        # This operation is virtually free in terms of memory. It just tells Python
        # to use the method resolution order (like finding .forward()) of the new class.
        base_model.__class__ = cls
        
        # logger.info("In-place upgrade complete. The object's class has been swapped.")
        
        # Step 3: Return the now-modified object.
        # Note that this is the *same object* that was passed in.
        return base_model
    
    def set_visual(self, visual):
        self.visual = visual

    def get_image_features(self, 
        pixel_values: torch.FloatTensor, 
        pixel_values_dinov2: Optional[torch.Tensor] = None,
        pixel_values_convnext: Optional[torch.Tensor] = None,
        pixel_values_pe_3d: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)

        if pixel_values_dinov2 is not None:
            pixel_values_dinov2 = pixel_values_dinov2.type(self.visual.dtype)
        if pixel_values_convnext is not None:
            pixel_values_convnext = pixel_values_convnext.type(self.visual.dtype)     
        if pixel_values_pe_3d is not None:
            pixel_values_pe_3d = pixel_values_pe_3d.type(self.visual.dtype)

        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values, 
            grid_thw=image_grid_thw, 
            pixel_values_dinov2=pixel_values_dinov2,
            pixel_values_convnext=pixel_values_convnext,
            pixel_values_pe_3d = pixel_values_pe_3d,
        )
        
        # import ipdb; ipdb.set_trace()
        # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        split_sizes = (image_grid_thw.prod(-1) // self.visual.config.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        pixel_values_dinov2: Optional[torch.Tensor] = None,
        pixel_values_convnext: Optional[torch.Tensor] = None,
        pixel_values_pe_3d: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            # image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values, 
                pixel_values_dinov2,
                pixel_values_convnext,
                pixel_values_pe_3d,
                image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )
    

class MultiEnc_Qwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    A custom Qwen3-VL model that uses two separate vision encoders.
    It inherits all functionality from the original model and overrides
    the __init__ and forward methods to support a second image input stream.
    """
    def __init__(self, 
        config, 
    ):
        super().__init__(config)

    def set_model(self, model):
        self.model = model

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        pixel_values_dinov2: Optional[torch.Tensor] = None,
        pixel_values_convnext: Optional[torch.Tensor] = None,
        pixel_values_pe_3d: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        >>> model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """
        # import ipdb; ipdb.set_trace()

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            pixel_values_dinov2=pixel_values_dinov2,
            pixel_values_convnext=pixel_values_convnext,
            pixel_values_pe_3d=pixel_values_pe_3d,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )