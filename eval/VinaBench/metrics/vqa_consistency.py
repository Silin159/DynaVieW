import os
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from time import sleep

from evaluators import GPTConsistencyEvaluator
        
def main():
    parser = argparse.ArgumentParser(description="VQA Consistency Evaluation")
    parser.add_argument("--dataset-name", type=str, default="vwp", help="Dataset to evaluate")
    parser.add_argument("--annotation-path", type=str, required=True, help="the path to dataset that includes transition narratives, ground truth images, and annotations")
    parser.add_argument("--model-name", type=str, required=True, help="image generation model")
    parser.add_argument("--judge-model", type=str, default="gpt-4o", help="Judge model to use")
    parser.add_argument("--eval-type", type=str, default="char", choices=["char", "location", "style"], help="Type of evaluation")
    parser.add_argument("--save-path", type=str, default="./results/consistency", help="the path to save evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="random seed used in generation")
    args = parser.parse_args()

    # set up evaluator
    evaluator = GPTConsistencyEvaluator(model=args.judge_model)

    # load samples and annotations
    with open(args.annotation_path, "r") as f:
        data_annotations = json.load(f)

    generated_image_folder = f"../results/seed_{args.seed}/{args.dataset_name}/"

    # perform evaluation
    for sample in tqdm(data_annotations):
        video_id = sample["scene_full_id"]
        story_id = sample["story_id"]

        generated_image_path = generated_image_folder + f"{video_id}/{story_id}/{args.model_name}"
        print(f"The {args.eval_type} consistency evaluation on sample {video_id}/{story_id} starts.")

        judge_folder_path = f"{generated_image_path}/{args.judge_model}-as-judge"
        os.makedirs(judge_folder_path, exist_ok=True)

        txt_list = sample["narrative"][1:]
        img_list = [os.path.join(generated_image_path, f"state_pred_{i}t{i+1}.jpg") for i in range(len(txt_list))]

        # pass the sample if its refined generation process is not complete.
        if not Path(f"{img_list[-1]}").exists():
            print(f"The refined generation on sample {video_id}/{story_id} is not complete, so skip it.")
            continue

        # pass if evaluation is already done on this sample
        save_dict_name = f"{judge_folder_path}/{args.eval_type}_consistency_score.json"
        if os.path.exists(save_dict_name):
            print(f"The evaluation on sample {video_id}/{story_id} with {args.judge_model} as the judge has done before, so skip it.")
            continue

        # get sample-wise annotations
        # skip the first narrative
        sp_char_desc = sample["global_profile"]
        sp_char_pres = [x["present"] for x in sample["scene_characters"]]
        locations = sample["location"]
        assert len(sp_char_desc) > 0, "No character descriptions found."

        vanilla_image_paths = [f"{generated_image_path}/state_0.jpg"]
        vanilla_image_paths += img_list

        # sleep(1)
        if args.eval_type == "char":
            print("Evaluating Character Consistency...")
            vanilla_consist_scores = evaluator.evaluate_character(
                image_paths=vanilla_image_paths,
                char_precense=sp_char_pres,
                char_description=sp_char_desc,
            )
        elif args.eval_type == "location":
            print("Evaluating Location Consistency...")
            vanilla_consist_scores = evaluator.evaluate_location(
                image_paths=vanilla_image_paths,
                locations=locations,
            )
        elif args.eval_type == "style":
            print("Evaluating Style Consistency...")
            vanilla_consist_scores = evaluator.evaluate_style(
                image_paths=vanilla_image_paths,
            )

        # save per-sample evaluation results
        save_dict_eval_scores = {
            "vanilla": vanilla_consist_scores,
        }
        with open(save_dict_name, "w") as f:
            json.dump(save_dict_eval_scores, f, indent=4)

if __name__ == "__main__":
    main()
