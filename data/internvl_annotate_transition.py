import os
import torch
import json
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
from prompts_schema import state_cap_instruct, state_cap_schema
from prompts_schema import state_trasition_instruct, state_trasition_cap, state_trasition_2nd_frame, state_trasition_schema

# for multi-job parallel running
split = 0
total_job = 1

# this is also the path to save the annotated transitions between keyframes
kf_root = "<your_path_to_the_saved_extracted_keyframes>"

full_img_pair_ls = []
for portion in ["Ego4D", "AgiBotWorld", "ShareGPT4Video"]:
    for kf_pth in os.listdir(os.path.join(kf_root, portion)):
        num_kf = 0
        for file in os.listdir(os.path.join(kf_pth, "keyframes")):
            if file.endswith(".jpg"):
                num_kf += 1
        for idx in range(num_kf-1):
            l_fid = f"{idx:03}"+".jpg"
            r_fid = f"{(idx+1):03}"+".jpg"
            l_img = os.path.join(os.path.join(kf_pth, "keyframes"), l_fid)
            r_img = os.path.join(os.path.join(kf_pth, "keyframes"), r_fid)
            full_img_pair_ls.append([l_img, r_img])
total_pair = len(full_img_pair_ls)
print(f"Total Image Pairs: {total_pair}", flush=True)

chunk = int(total_pair/total_job)+1
left = split*chunk
right = min((split+1)*chunk, total_pair)
target_pairs = full_img_pair_ls[left:right]


model_path = '<your_path_to_the_downloaded_internvl_model>'
batch_size = 16
llm = LLM(model=model_path, tensor_parallel_size=4, dtype=torch.bfloat16, max_num_seqs=batch_size, max_model_len=32768,
          trust_remote_code=True, limit_mm_per_prompt={"image": 2}, mm_processor_kwargs={"max_dynamic_patch": 6})
tokenizer = AutoTokenizer.from_pretrained(model_path)

stop_tokens = ["<|endoftext|>", "<|im_end|>"]
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

cap_guided = GuidedDecodingParams(json=state_cap_schema)
cap_sampling_params = SamplingParams(temperature=0.0, max_tokens=16384, stop_token_ids=stop_token_ids, guided_decoding=cap_guided)
cap_msg = [{"role": "user", "content": f"""{state_cap_instruct}\n<image>\n"""}]
cap_prompt = tokenizer.apply_chat_template(cap_msg, tokenize=False, add_generation_prompt=True)

trans_guided = GuidedDecodingParams(json=state_trasition_schema)
trans_sampling_params = SamplingParams(temperature=0.0, max_tokens=16384, stop_token_ids=stop_token_ids, guided_decoding=trans_guided)


def parse_json(json_str):
    try:
        out_json = eval("{"+json_str.split("{",1)[1].rsplit("}",1)[0]+"}")
        return True, out_json
    except:
        print("Parse JSON Fail!", flush=True)
        return False, json_str


print(f"Start Reading Portion Images", flush=True)
img_pils = []
for pair in target_pairs:
    img_pils.append(
        [Image.open(pair[0]).convert('RGB'), Image.open(pair[1]).convert('RGB')]
    )
print(f"Finish Reading Portion Images", flush=True)

for start in range(0, len(target_pairs), batch_size):
    end = min(start+batch_size, len(target_pairs))
    idx_map = list(range(start, end))
    
    print(f"Start Split {split} Batch: {start} to {end-1}", flush=True)
    
    trans_files = [pair[0].replace("/keyframes/", "/transitions/").replace(".jpg", ".json") for pair in target_pairs[start:end]]
    if all([os.path.exists(file) for file in trans_files]):
        print(f"Already Done Split {split} Batch: {start} to {end-1}", flush=True)
        continue

    st_trans = []
    
    mm_input_cap = []
    for idx in idx_map:
        st_trans.append({"state": "", "transition": ""})
        mm_input_cap.append(
            {"prompt": cap_prompt, "multi_modal_data": {"image": img_pils[idx][0]}}
        )
    out_resp = llm.generate(mm_input_cap, sampling_params=cap_sampling_params)
    
    for oid, out in enumerate(out_resp):
        out_txt = out.outputs[0].text
        success, out_json = parse_json(out_txt)
        out_cap = out_json.get("caption", str(out_json)) if success else str(out_json)
        st_trans[oid]["state"] = out_cap
    
    mm_input_trans = []
    for oid, idx in enumerate(idx_map):
        out_cap = st_trans[oid]["state"]
        trans_msg = [{"role": "user", "content": f"""{state_trasition_instruct}\n<image>\n{state_trasition_cap} {out_cap}\n\n{state_trasition_2nd_frame}\n<image>\n"""}]
        trans_prompt = tokenizer.apply_chat_template(trans_msg, tokenize=False, add_generation_prompt=True)
        mm_input_trans.append(
            {"prompt": trans_prompt, "multi_modal_data": {"image": [img_pils[idx][0], img_pils[idx][1]]}}
        )
    out_resp = llm.generate(mm_input_trans, sampling_params=trans_sampling_params)

    for oid, out in enumerate(out_resp):
        out_txt = out.outputs[0].text
        _, out_json = parse_json(out_txt)
        st_trans[oid]["transition"] = str(out_json)

    for oid, idx in enumerate(idx_map):
        trans_file = target_pairs[idx][0].replace("/keyframes/", "/transitions/").replace(".jpg", ".json")
        os.makedirs(os.path.dirname(trans_file), exist_ok=True)
        with open(trans_file, "w") as f:
            json.dump(st_trans[oid], f, indent=2)
    
    print(f"Done Split {split} Batch: {start} to {end-1}", flush=True)
