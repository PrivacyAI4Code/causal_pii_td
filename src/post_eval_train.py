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
    summary_file = os.path.join(args.output_dir, "eval_train_summary.json")
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
            f.write(f"{ckpt_name} - train accuracy: {results['accuracy']} correct: {results['correct_tokens']} total: {results['total_tokens']}\n")
        

if __name__ == "__main__":
    main()