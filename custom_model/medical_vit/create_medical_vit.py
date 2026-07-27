import torch

# Import the necessary components from our new local files
from .mae_v2 import vit_large
from .pe import MyCLIP_ssl

def create_ssl_large_p8_3d_pe_model(ckpt_path, patch_size = None, return_config = False):
    """
    Creates and loads the 'SSL_large_p8_3d_pe' model.
    """
    print("Creating vision encoder: vit_large_patch8 for 3D...")
    vision_encoder, config_dict = vit_large(
        image_size=128,  # This size is specified to correctly load the 3D absolute positional embedding
        patch_size=8 if patch_size is None else patch_size,
        use_rope=True,
        drop_path=0.2,
        depth_patch_size=8 if patch_size is None else patch_size,
        # pool_type='attn',
        pool_type='none',
        return_config=return_config,
    )

    
    # try:
    #     ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    # except FileNotFoundError:
    #     print(f"ERROR: Checkpoint file not found at '{ckpt_path}'.")

    # loading_info = vision_encoder.load_state_dict(ckpt, strict=True)

    # --- Combine into the final model ---
    print("Put vision encoder into MyCLIP_ssl model...")
    model = MyCLIP_ssl(vision_encoder=vision_encoder, text_encoder=None)

    # --------- Load pre-trained weights ---------
    if ckpt_path is not None and ckpt_path != '':
        load_ckpt(model, ckpt_path)

    return model, config_dict

    # # --- IMPORTANT ---
    # # Change this path to where you have saved the checkpoint file.
    # ckpt_path = '/mnt/eff_nas/hanyizeng/code/Med-CLIP3D/ckpts/ssl_large_p8_320k.pt'
    # print(f"Loading vision encoder checkpoint from: {ckpt_path}")
    
    # try:
    #     ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    # except FileNotFoundError:
    #     print(f"ERROR: Checkpoint file not found at '{ckpt_path}'.")
    #     print("Please download the 'ssl_large_p8_320k.pt' file and update the 'ckpt_path' variable in main.py.")
    #     return None, None

    # # Clean the checkpoint state dictionary
    # new_state_dict = OrderedDict()
    # for k, v in ckpt.items():
    #     name = k.replace('module.', '', 1)  # More robust way to remove prefix
    #     new_state_dict[name] = v

    # # Remove keys that are not part of the encoder
    # new_state_dict.pop('pos_embed', None)
    # new_state_dict.pop('mask_token', None)
    # filtered_state_dict = {k: v for k, v in new_state_dict.items() if not k.startswith('decoder')}

    # print("Loading state dictionary into the vision encoder...")
    # loading_info = vision_encoder.load_state_dict(filtered_state_dict, strict=False)
    # print("Vision Encoder Loading Info:", loading_info)

    # # --- Create the Text Encoder ---
    # print("Creating text encoder: BiomedVLP-CXR-BERT-specialized...")
    # tokenizer_path = 'ckpts/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc'
    # try:
    #     tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True, local_files_only=True)
    #     text_encoder = BertModel.from_pretrained(tokenizer_path)
    # except OSError:
    #     print(f"ERROR: Text model/tokenizer files not found in '{tokenizer_path}'")
    #     print("Please make sure you have downloaded the BiomedVLP-CXR-BERT-specialized model into the 'ckpts' directory.")
    #     return None, None
        
    # text_encoder.resize_token_embeddings(len(tokenizer))

    # # --- Combine into the final model ---
    # print("Combining vision and text encoders into MyCLIP_ssl model...")
    # model = MyCLIP_ssl(vision_encoder=vision_encoder, text_encoder=text_encoder)

    # return model, tokenizer


def load_ckpt(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if 'epoch' in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        loading_info = model.load_state_dict(sd, strict = False)
        assert len(loading_info.missing_keys) == 0, "We must assert the missing keys to be None to ensure a complete param loading."
        
        print(f"=> Loaded checkpoint '{ckpt_path}' (epoch {start_epoch}): {loading_info}")
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        loading_info = model.load_state_dict(checkpoint)
        print(f"=> Loaded checkpoint '{ckpt_path}': {loading_info}")


# Example of how to use the function
if __name__ == "__main__":
    print("--- Running Model Creation ---")
    model, tokenizer = create_ssl_large_p8_3d_pe_model()

    if model and tokenizer:
        print("\n--- Model and Tokenizer created successfully! ---")
        print("\nFinal Model Architecture:")
        print(model)
        # You can now use the 'model' and 'tokenizer' for your tasks.
