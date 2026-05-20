# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math
import random
from PIL import Image

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    # image = torch.einsum("chpwq->hwpqc", image)
    # chpwq->hcpwq->hwpcq->hwpqc
    image = image.transpose(0, 1).transpose(1, 3).transpose(3, 4)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within 
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    space_token = tokenizer.convert_tokens_to_ids('Ċ')

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image, 
        space_token=space_token,
    )

    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def parse_json_schema(text):
    if text.startswith("{") and text.endswith("}"):
        json_txt = eval(text)
        for key, value in json_txt.items():
            if isinstance(value, str):
                json_txt[key] = ''
            if isinstance(value, list):
                json_txt[key] = []
                for item in value:
                    if isinstance(item, str):
                        json_txt[key].append('')
                    if isinstance(item, dict):
                        json_txt[key].append(parse_json_schema(str(item)))
            if isinstance(value, dict):
                json_txt[key] = parse_json_schema(str(value))
        return json_txt
    else:
        return None


def schema2weight(x, schema=None):
    # token_weight = 1 / (len(x) ** 0.5)
    # schema_weight = 0.1 / (len(schema) ** 0.5)
    token_weight = 1.0
    schema_weight = 0.1
    if schema is None:
        return [token_weight] * (len(x)+1)
    weights = []
    sidx = 0
    for tid in x:
        if sidx < len(schema) and tid == schema[sidx]:
            weights.append(schema_weight)
            sidx += 1
        else:
            weights.append(token_weight)
    weights.append(token_weight)  # eos token
    return weights


def prepare_attention_mask_vwm(split_lens, attn_modes, device="cpu", img_hist_dropout=0.3, vit_attn_trans=True, vwm_attn=True):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within 
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    prev_prompt = []
    prev_trans = []
    prev_visual = []
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ["causal", "full", "noise", "causal_trans", "causal_prompt", "full_state", "noise_state"]
        if "causal" in attn_mode:
            attention_mask[csum:csum+s, csum:csum+s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum+s, :csum] = 1
            if vwm_attn and ("trans" in attn_mode or "prompt" in attn_mode):
                for prc in prev_trans:
                    attention_mask[csum:csum+s, prc[0]:prc[1]] = 0
                for vid, prv in enumerate(prev_visual):
                    if vid < len(prev_visual)-1 and random.random() < img_hist_dropout:
                        attention_mask[csum:csum+s, prv[0]:prv[1]] = 0
                
                if "trans" in attn_mode:
                    prev_trans.append([csum, csum+s])
                if "prompt" in attn_mode:
                    prev_prompt.append([csum, csum+s])
        else:
            attention_mask[csum:csum+s, csum:csum+s] = torch.ones((s, s))
            if vwm_attn and "state" in attn_mode:
                if len(prev_trans) > 0 and ("noise" in attn_mode or vit_attn_trans):
                    lat = prev_trans[-1]
                    attention_mask[csum:csum+s, lat[0]:lat[1]] = 1
                # if len(prev_visual) > 0:
                #     lav = prev_visual[-1]
                #     attention_mask[csum:csum+s, lav[0]:lav[1]] = 1
                for vid, prv in enumerate(prev_visual):
                    if vid == len(prev_visual)-1 or random.random() >= img_hist_dropout:
                        attention_mask[csum:csum+s, prv[0]:prv[1]] = 1
                for pmpt in prev_prompt:
                    attention_mask[csum:csum+s, pmpt[0]:pmpt[1]] = 1
            else:
                attention_mask[csum:csum+s, :csum] = 1
            if "noise" not in attn_mode:
                prev_visual.append([csum, csum+s])
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if "noise" in attn_mode:
            attention_mask[:, csum:csum+s] = torch.zeros((sample_len, s))
            attention_mask[csum:csum+s, csum:csum+s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask
