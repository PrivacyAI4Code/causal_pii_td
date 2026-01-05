#!/usr/bin/env python
"""
post_eval on test and dynamic datasets, 
output the learning dynamics of the model
"""

import os
# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Tuple
import argparse
import yaml
from pathlib import Path
from transformers import HfArgumentParser, set_seed
import numpy as np
import torch
import matplotlib.pyplot as plt

from post_func import ModelArguments, DataArguments, evaluate_model_on_dataset, build_checkpoint_paths, load_model_and_tokenizer, load_dataset_for_post
from utils import log_print, flatten_config

def load_args() -> argparse.Namespace:
    """Parse CLI arguments for post-evaluation workflow."""
    parser = argparse.ArgumentParser(description="Post-evaluation for fine-tuned models")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--test_file", type=str, required=True, help="path to test file")
    parser.add_argument("--dynamic_file", type=str, required=True, help="path to dynamic file")
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
    summary_file = os.path.join(args.output_dir, "eval_summary.json")
    if os.path.exists(summary_file):
        os.remove(summary_file)

    last_tokenizer = None
    base_model = None
    base_tokenizer = None

    all_detailed_predictions = []
    fig_data = {}

    for idx, (ckpt_name, checkpoint_path) in enumerate(checkpoint_paths):
        log_print(f"\n=== Evaluating checkpoint: {ckpt_name} ===")
        if idx == 0:
            model, tokenizer = load_model_and_tokenizer(checkpoint_path, model_args, data_args, using_lora=False, using_fim=data_args.using_fim, base_model=base_model, base_tokenizer=base_tokenizer)
            base_model = model
            base_tokenizer = tokenizer
        else:
            model, tokenizer = load_model_and_tokenizer(checkpoint_path, model_args, data_args, using_lora=True, using_fim=data_args.using_fim, base_model=base_model, base_tokenizer=base_tokenizer)
        
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        if tokenizer != last_tokenizer:
            new_tokenizer_flag = True
            log_print(f"Tokenizer changed, preprocessing dataset...")
        else:
            new_tokenizer_flag = False
        last_tokenizer = tokenizer

        if new_tokenizer_flag:
            _test_dataset = load_dataset_for_post(args.test_file, data_args, block_size, tokenizer, mode="test", verbose=True, np_rng=np_rng)
        results, _ = evaluate_model_on_dataset(model, tokenizer, _test_dataset, data_args, device=device, mode="test", np_rng=np_rng)

        with open(summary_file, "a") as f:
            f.write(f"{ckpt_name} - Test accuracy: {results['accuracy']} correct: {results['correct_tokens']} total: {results['total_tokens']}\n")
        
        if new_tokenizer_flag:
            _dynamic_dataset = load_dataset_for_post(args.dynamic_file, data_args, block_size, tokenizer, mode="dynamic", verbose=True, np_rng=np_rng)
        results, detailed_predictions = evaluate_model_on_dataset(model, tokenizer, _dynamic_dataset, data_args, device=device, mode="dynamic", np_rng=np_rng)

        # show the top 3 detailed predictions for each type when idx == 1
        if idx == 1:
            # Group detailed_predictions by pii_type
            from collections import defaultdict
            type_to_preds = defaultdict(list)
            for pred in detailed_predictions:
                pii_type = pred.get("type", None)
                if pii_type is not None:
                    type_to_preds[pii_type].append(pred)
            log_print("Top 3 detailed predictions for each pii_type (by mean pred_token_probs):")
            for pii_type, preds in type_to_preds.items():
                log_print(f"{pii_type}: ")
                for pred in preds[:3]:
                    log_print(f"  {pred}")
        
        all_detailed_predictions.extend(detailed_predictions)
    
    dynamic_dataset = json.load(open(args.dynamic_file, "r"))
    # regroup by example_id
    avg_pred_token_probs_by_example_id = {}
    for prediction in all_detailed_predictions:
        example_id = prediction["example_id"]
        if example_id not in avg_pred_token_probs_by_example_id:
            avg_pred_token_probs_by_example_id[example_id] = []
        avg_pred_token_probs_by_example_id[example_id].append(np.mean(prediction["pred_token_probs"]))
    
    confidences_by_type = {}
    variabilities_by_type = {}
    for example in dynamic_dataset:
        example_id = example["id"]
        example_type = example["piiType"]
        if example_id not in avg_pred_token_probs_by_example_id:
            raise ValueError(f"example_id {example_id} not found in avg_pred_token_probs_by_example_id")
        
        confidence = np.mean(avg_pred_token_probs_by_example_id[example_id])
        variability = np.std(avg_pred_token_probs_by_example_id[example_id])
        if example_type not in confidences_by_type:
            confidences_by_type[example_type] = []
            variabilities_by_type[example_type] = []
        confidences_by_type[example_type].append(confidence)
        variabilities_by_type[example_type].append(variability)
        example["confidence"] = confidence
        example["variability"] = variability
    
    fig_data["confidences_by_type"] = confidences_by_type
    fig_data["variabilities_by_type"] = variabilities_by_type
            
    vis_dir = os.path.join(args.output_dir, config["dataset"]["visualize_dir"])
    os.makedirs(vis_dir, exist_ok=True)
    for type in confidences_by_type.keys():
        plt.figure(figsize=(8, 6))
        plt.scatter(variabilities_by_type[type], confidences_by_type[type], alpha=0.7)
        plt.xlabel("Variability")
        plt.ylabel("Confidence")
        plt.title(f"Confidence vs Variability for type: {type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"confidence_variability_{type}.png"))
        plt.close()
        log_print(f"Saved visualization for type {type} to {os.path.join(vis_dir, f'confidence_variability_{type}.png')}")
    
    # draw two histograms, one is confidence, one is variability, x axis is type, y axis is value
    types = sorted(confidences_by_type.keys())  
    conf_means = [np.mean(confidences_by_type[t]) for t in types]
    var_means  = [np.mean(variabilities_by_type[t]) for t in types]

    fig_data["conf_means"] = conf_means
    fig_data["var_means"] = var_means

    # --------------------------
    # 图1: Confidence 均值
    # --------------------------
    plt.figure(figsize=(8, 6))
    plt.bar(types, conf_means, color="skyblue")
    plt.ylabel("Mean Confidence")
    plt.title("Mean Confidence by Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"confidence_histogram.png"))
    plt.close()
    log_print(f"Saved visualization for confidence histogram to {os.path.join(vis_dir, f'confidence_histogram.png')}")

    # --------------------------
    # 图2: Variability 均值
    # --------------------------
    plt.figure(figsize=(8, 6))
    plt.bar(types, var_means, color="salmon")
    plt.ylabel("Mean Variability")
    plt.title("Mean Variability by Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"variability_histogram.png"))
    plt.close()
    log_print(f"Saved visualization for variability histogram to {os.path.join(vis_dir, f'variability_histogram.png')}")
    
    with open(os.path.join(args.output_dir, f"post_eval_dynamic_dataset.json"), "w") as f:
        json.dump(dynamic_dataset, f, indent=2)
    log_print(f"Saved dynamic dataset to {os.path.join(args.output_dir, f'dynamic_dataset.json')}")

    with open(os.path.join(args.output_dir, f"post_eval_fig_data.json"), "w") as f:
        json.dump(fig_data, f, indent=2)
    log_print(f"Saved fig data to {os.path.join(args.output_dir, f'post_eval_fig_data.json')}")

    with open(os.path.join(args.output_dir, f"post_eval_all_detailed_predictions_clean.json"), "w") as f:
        json.dump(all_detailed_predictions, f, indent=2)
    log_print(f"Saved all detailed predictions to {os.path.join(args.output_dir, f'all_detailed_predictions_clean.json')}")

if __name__ == "__main__":
    main()