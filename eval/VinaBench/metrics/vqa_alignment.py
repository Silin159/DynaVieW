import os
import json
import numpy as np
import base64
from tqdm import tqdm
import argparse
from pathlib import Path
from time import sleep

from evaluators import GPTAlignmentEvaluator

def main():
    parser = argparse.ArgumentParser(description="VQA Alignment Evaluation")
    parser.add_argument("--dataset-name", type=str, default="vwp", help="Dataset to evaluate")
    parser.add_argument("--annotation-path", type=str, required=True, help="the path to dataset that includes transition narratives, ground truth images, and annotations")
    parser.add_argument("--model-name", type=str, required=True, help="image generation model")
    parser.add_argument("--judge-model", type=str, default="gpt-4o", help="Judge model to use")
    parser.add_argument("--eval-type", type=str, default="char_num", choices=["char_num", "char_attr", "entity", "location", "time"], help="Type of evaluation")
    parser.add_argument("--save-path", type=str, default="./results/alignment", help="the path to save evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="random seed used in generation")
    args = parser.parse_args()

    # set up evaluator
    evaluator = GPTAlignmentEvaluator(model=args.judge_model)

    # load samples and annotations
    with open(args.annotation_path, "r") as f:
        data_annotations = json.load(f)

    generated_image_folder = f"../results/seed_{args.seed}/{args.dataset_name}/"

    # perform evaluation
    for sample in tqdm(data_annotations):
        video_id = sample["scene_full_id"]
        story_id = sample["story_id"]

        generated_image_path = generated_image_folder + f"{video_id}/{story_id}/{args.model_name}"
        print(f"The {args.eval_type} alignment evaluation on sample {video_id}/{story_id} starts.")

        judge_folder_path = f"{generated_image_path}/{args.judge_model}-as-judge"
        os.makedirs(judge_folder_path, exist_ok=True)

        txt_list = sample["narrative"][1:]
        img_list = [os.path.join(generated_image_path, f"state_pred_{i}t{i+1}.jpg") for i in range(len(txt_list))]
        
        # pass the sample if its refined generation process is not complete.
        if not Path(f"{img_list[-1]}").exists():
            print(f"The refined generation on sample {video_id}/{story_id} is not complete, so skip it.")
            continue

        # pass if evaluation is already done on this sample
        save_dict_name = f"{judge_folder_path}/{args.eval_type}_alignment_score.json"
        if os.path.exists(save_dict_name):
            print(f"The evaluation on sample {video_id}/{story_id} with {args.judge_model} as the judge has done before, so skip it.")
            continue

        per_sample_eval_scores = {
            "vanilla": [],
        }

        # get sample-wise annotations
        # skip the first narrative
        sp_char_desc = sample["global_profile"]
        sp_char_pres = [x["present"] for x in sample["scene_characters"]][1:]
        sp_char_num = [int(x["num_present"]) for x in sample["scene_characters"]][1:]
        entities = [linked_ent.split("; ") for linked_ent in sample["linked_entities"]][1:]
        locations = sample["location"][1:]
        times = sample["time"][1:]
        assert len(sp_char_desc) > 0, "No character descriptions found."

        # for id, plot in enumerate(target_scene_dict["narrative"][1:]):
        for idx in range(len(img_list)):
            print(f"Evaluating image scene {idx+1} ...")
            # get character presence and description for this turn
            turn_char_pres = sp_char_pres[idx]
            char_count = sp_char_num[idx]
            turn_char_desc = {}
            for char in turn_char_pres:
                turn_char_desc[char] = sp_char_desc[char]
            # get entities
            entity = entities[idx]
            if args.eval_type == "entity":
                if entity == ['']:
                    print("  No entities linked for this scene, skipping ...")
                    continue
                elif all([ent[0].isupper() for ent in entity]):
                    print("  No non-character entities linked for this scene, skipping ...")
                    continue
            # get location
            loc = locations[idx]
            # get time
            time = times[idx]
            if args.eval_type == "time" and time == "unclear":
                print("  Time is unclear for this scene, skipping ...")
                continue

            # evaluate each transition twice, before and after feedback
            vanilla_image_path = img_list[idx]
            print(vanilla_image_path)

            # sleep(1)
            if args.eval_type == "char_num":
                vanilla_res = evaluator.evaluate_character_num(image_path=vanilla_image_path, true_char_num=char_count)
            elif args.eval_type == "char_attr":
                vanilla_res = evaluator.evaluate_character_attr(image_path=vanilla_image_path, char_desc=turn_char_desc)
            elif args.eval_type == "entity":
                vanilla_res = evaluator.evaluate_entities(image_path=vanilla_image_path, entities=entity)
            elif args.eval_type == "location":
                vanilla_res = evaluator.evaluate_location(image_path=vanilla_image_path, location=loc)
            elif args.eval_type == "time":
                vanilla_res = evaluator.evaluate_time(image_path=vanilla_image_path, time=time)
            else:
                raise ValueError(f"Unknown eval type: {args.eval_type}")
            
            if vanilla_res is not None:
                per_sample_eval_scores["vanilla"].append(vanilla_res)

        # save per-sample evaluation results
        save_dict_eval_scores = {
            "vanilla": per_sample_eval_scores["vanilla"],
        }
        with open(save_dict_name, "w") as f:
            json.dump(save_dict_eval_scores, f, indent=4)

if __name__ == "__main__":
    main()
