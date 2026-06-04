import os
import sys
from pathlib import Path

# Run from ./eval/LEGO: add ../modeling so data/, modeling/, inferencer_vwm resolve like BAGEL.
_MODELING_ROOT = Path(__file__).resolve().parent.parent.parent / "modeling"
if str(_MODELING_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODELING_ROOT))

from copy import deepcopy
import random
import numpy as np
import json
import ast
import argparse
from tqdm import tqdm

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from inferencer_vwm import InterleaveInferencer

def split_dict(d: dict, n_slices: int):
    """
    Split dictionary `d` into `n_slices` smaller dictionaries.
    """
    items = list(d.items())
    chunk_size = (len(items) + n_slices - 1) // n_slices  # ceil

    return [
        dict(items[i * chunk_size:(i + 1) * chunk_size])
        for i in range(n_slices)
    ]

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="World Model Inference")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--model_name", type=str, help="the name of the model to run")
    parser.add_argument("--dataset_name", type=str, help="the name of the dataset to run")
    parser.add_argument("--checkpoint", type=str, help="name of the checkpoint to run")
    args = parser.parse_args()

    # replace <...> with your root path
    offload_path = "..."  # create an empty offload folder
    config_path = "..."  # path for loading BAGEL config and VAE module
    output_root = f"..."  # path for saving the output

    # our pre-trained vision world model checkpoints
    pt_model_root = "..."  # path for loading the pre-trained vision world model checkpoints
    model_name = args.model_name
    checkpoint = args.checkpoint
    model_path = os.path.join(os.path.join(pt_model_root, model_name), checkpoint)

    llm_config = Qwen2Config.from_json_file(os.path.join(config_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(config_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(config_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(config_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(518, 224, 14)

    max_mem_per_gpu = "80GiB"
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print(device_map)
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "model.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=offload_path,
    )
    model = model.eval()
    print('Model loaded')

    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # vwm_mask=True: use our selective attention mask
    # transition_only=True: only predict the next transition (text)
    # state_only=True: only predict the next state (image)
    if args.dataset_name == "ego4d":
        root_folder = ".../lego/EgoGen/ego4d.fho/val"
        gen_folder = f".../lego/EgoGen/ego4d.fho/gen_vwm_sft_{args.checkpoint}"
        prompt_template = "Please predict the next egocentric frame based on the following transition, where '#C' refers to the camera-wearer:\n{}"
    elif args.dataset_name == "epickitchen":
        root_folder = ".../lego/EgoGen/epickitchen/val"
        gen_folder = f".../lego/EgoGen/epickitchen/gen_vwm_sft_{args.checkpoint}"
        prompt_template = "Please predict the next egocentric frame based on the following transition:\n{}"

    with open((f".../lego/{args.dataset_name}_metadata.json"), "r") as f:
        metadata = json.load(f)

    # load your input sequences of interleaved states (images) and transitions (texts)
    with open(f".../lego/{args.dataset_name}_val.json", "r") as f:
        annotations = json.load(f)

    for clip_id, clip_id_sample in tqdm(annotations.items()):
        width = metadata[clip_id]["width"]
        height = metadata[clip_id]["height"]
        inference_hyper=dict(
            vwm_mask=True,
            transition_only=False,
            state_only=True,
            max_text_tokens=2048,
            do_sample=False,
            text_temperature=0.0,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.8, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=(256, width / height * 256), # actually it's not gonna work since the model always uses a default resize transform to adjust the size to align with pretraining.
            enable_taylorseer=False,
            img_dropout_trans=0.0,
            img_dropout_state=0.0,
            complete=False,
        )
        for action_id, action_id_sample in clip_id_sample.items():
            input_img = Image.open(os.path.join(root_folder, action_id_sample["image_0"]))
            narration = action_id_sample["action"]

            output_dir = os.path.join(gen_folder, clip_id, action_id)
            os.makedirs(output_dir, exist_ok=True)


            if os.path.exists(os.path.join(gen_folder, action_id_sample["image_1"])):
                print(f"Skipping {clip_id} action {action_id} as output already exists.")
                continue
            
            input_lists = [input_img, prompt_template.format(narration)]
            print(prompt_template.format(narration))
            input_types = ["image", "prompt"]
            # The context only contains states and narratives. 
            output_dict = inferencer(input_list=input_lists, input_types=input_types, **inference_hyper)

            output_dict["image"].save(os.path.join(gen_folder, action_id_sample["image_1"]))

if __name__ == "__main__":
    main()