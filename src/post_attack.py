#!/usr/bin/env python
"""
pii attack on the model, using the prefix or mix to attack the model
"""
import os
# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import yaml
from pathlib import Path
from transformers import HfArgumentParser, set_seed
import numpy as np
import torch
from match import batch_match

from post_func import ModelArguments, DataArguments, evaluate_model_on_dataset, build_checkpoint_paths, load_model_and_tokenizer, load_dataset_for_post
from utils import log_print, flatten_config

def load_args() -> argparse.Namespace:
    """Parse CLI arguments for post-evaluation workflow."""
    parser = argparse.ArgumentParser(description="Post-evaluation for fine-tuned models")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--attack_file", type=str, required=True, help="path to attack file")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory for evaluation results")
    args = parser.parse_args()
    return args


def main():
    args = load_args()
    log_print(f"Running post-evaluation with args: {args}")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    flat_config = flatten_config(config)
    merged_args = {**flat_config, **vars(args)}

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(merged_args, allow_extra_keys=True)
    

    log_print(f"model_args: {model_args}")
    log_print(f"data_args: {data_args}")

    seed_num = data_args.seed
    set_seed(seed_num)
    os.makedirs(args.output_dir, exist_ok=True)

    np_rng = np.random.RandomState(seed=seed_num)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_paths = build_checkpoint_paths(model_args, data_args)
    summary_file = os.path.join(args.output_dir, "attack_summary.json")
    if os.path.exists(summary_file):
        os.remove(summary_file)

    last_tokenizer = None
    base_model = None
    base_tokenizer = None

    all_detailed_predictions = []

    #for idx, (ckpt_name, checkpoint_path) in enumerate(checkpoint_paths):
    base_checkpoint_path = checkpoint_paths[0][1]
    target_checkpoint_path = checkpoint_paths[-1][1]
    base_model, base_tokenizer = load_model_and_tokenizer(base_checkpoint_path, model_args, data_args, using_lora=False, using_fim=data_args.using_fim, base_model=base_model, base_tokenizer=base_tokenizer, mode="attack")
    model, tokenizer = load_model_and_tokenizer(target_checkpoint_path, model_args, data_args, using_lora=True, using_fim=data_args.using_fim, base_model=base_model, base_tokenizer=base_tokenizer, mode="attack")
    log_print(f"\n=== Evaluating checkpoint: {target_checkpoint_path} ===")
    block_size = min(data_args.block_size, tokenizer.model_max_length)
    if tokenizer != last_tokenizer:
        new_tokenizer_flag = True
        log_print(f"Tokenizer changed, preprocessing dataset...")
    else:
        new_tokenizer_flag = False
    last_tokenizer = tokenizer

    if new_tokenizer_flag:
        _attack_dataset = load_dataset_for_post(args.attack_file, data_args, block_size, tokenizer, mode="attack", verbose=True, np_rng=np_rng)
        
    summary_info, detailed_predictions = evaluate_model_on_dataset(model, tokenizer, _attack_dataset, data_args, device=device, mode="attack", np_rng=np_rng)

    all_detailed_predictions.extend(detailed_predictions)

    # regroup by type 
    detailed_predictions_by_type = {}
    for prediction in all_detailed_predictions:
        pii_type = prediction["type"]
        if pii_type not in detailed_predictions_by_type:
            detailed_predictions_by_type[pii_type] = []
        detailed_predictions_by_type[pii_type].append(prediction)
    
    
    try:
        with open(args.attack_file, "r") as f:
            attack_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log_print(f"Error loading attack dataset from {args.attack_file}: {e}")
        raise

    results_by_type = {}
    hit_unique_id = []
    
    # for each type, organize the answer
    for pii_type in detailed_predictions_by_type.keys():
        answer = []
        generated_text = []
        unique_id = []
        for prediction in detailed_predictions_by_type[pii_type]:
            answer.append(prediction["expected_text"])
            generated_text.append(prediction["generated_text"])
            unique_id.append((prediction["example_id"], prediction["middle_start"], prediction["middle_start"] + len(prediction["expected_text"])))
        hit_count = 0
        all_count = 0
        which_hit = []
        matched = batch_match(generated_text, answer, ignore_case=False, return_all=True, parallel=False)
        for id_hit, hit in enumerate(matched):
            all_count += 1
            if hit:
                log_print(f"Found hit in {generated_text[id_hit]}")
                hit_count += 1
                which_hit.append(unique_id[id_hit])
                hit_unique_id.append(unique_id[id_hit])
        results_by_type[pii_type] = {"hit_count": hit_count, "all_count": all_count, "which_hit": which_hit}
    
    for example in attack_dataset:
        example["text"] = None
        if (example["id"], example["location_start"], example["location_end"]) in hit_unique_id:
            example["attack_hit"] = True
        else:
            example["attack_hit"] = False
                
    output_file = os.path.join(args.output_dir, "post_attack_dataset.json")
    try:
        with open(output_file, "w") as f:
            for example in attack_dataset:
                f.write(json.dumps(example) + "\n")
        log_print(f"Saved attack dataset to {output_file}")
    except IOError as e:
        log_print(f"Error saving attack dataset to {output_file}: {e}")
        raise

    # save all_detailed_predictions
    with open(os.path.join(args.output_dir, "post_attack_all_detailed_predictions.json"), "w") as f:
        json.dump(all_detailed_predictions, f, indent=2)
    log_print(f"Saved all_detailed_predictions to {os.path.join(args.output_dir, 'post_attack_all_detailed_predictions.json')}")

if __name__ == "__main__":
    main()