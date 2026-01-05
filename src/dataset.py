"""
This module contains methods for constructing the dataset for the PII related tasks.
"""

import argparse
import os
import yaml
from pathlib import Path
import hashlib
import json

from utils import save_json, save_jsonl, log_print, load_json_safe
from download import download_dataset
from search import search_pii
from preprocess import merge_clean_dataset
from refine_multiworker import refine_dataset
from sample import sample_dataset

def load_args():
    parser = argparse.ArgumentParser(description="loads config and runs the pipeline")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory for the dataset")
    args = parser.parse_args()
    return args

def main():
    args = load_args()
    log_print(f"running src/dataset.py with args: {args}")

    # Load the config from yaml
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["source_name"]
    subset = config["dataset"]["subset"]
    num_samples = config["dataset"]["num_samples"]
    num_refine_samples = config["dataset"]["num_refine_samples"]
    n_train = config["dataset"]["n_train"]
    n_val = config["dataset"]["n_val"]
    n_test = config["dataset"]["n_test"]
    sample_size = config["dataset"]["sample_size"]
    visualize_size = config["dataset"]["visualize_size"]
    visualize_dir = os.path.join(args.output_dir, config["dataset"]["visualize_dir"])

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(visualize_dir, exist_ok=True)

    context_length = config["dataset"]["refine_context_length"]
    sample_threshold = config["dataset"]["sample_threshold"]
    using_cache = config["dataset"]["using_cache"]

    not_using_cache = [k for k, v in using_cache.items() if not v]

    context_length_str = "_".join([f"{k}:{v}" for k, v in sorted(context_length.items())])
    sample_threshold_str = "_".join([f"{k}:{v}" for k, v in sorted(sample_threshold.items())])
    h = hashlib.md5((context_length_str + sample_threshold_str).encode()).hexdigest()

    # File paths
    download_dataset_file = os.path.join(output_dir, f"download_{dataset_name.replace('/', '_')}_{subset}_{num_samples}.json")
    search_dataset_file = os.path.join(output_dir, f"search_{dataset_name.replace('/', '_')}_{subset}_{num_samples}.json")
    merge_dataset_file = os.path.join(output_dir, f"merge_{dataset_name.replace('/', '_')}_{subset}_{num_samples}.json")
    refine_dataset_file = os.path.join(output_dir, f"refine_{dataset_name.replace('/', '_')}_{subset}_{num_samples}_{num_refine_samples}_{h}.json")
    cache_file = os.path.join(args.output_dir, f"cache_{dataset_name.replace('/', '_')}_{subset}.json") # should be under the data folder
    cache_file_backup = os.path.join(args.output_dir, f"cache_{dataset_name.replace('/', '_')}_{subset}_backup.json") # should be under the data folder
    #sample_dataset_file = os.path.join(output_dir, f"sample_{dataset_name.replace('/', '_')}_{subset}_{num_samples}.json")

    # 1. Download dataset
    if not os.path.exists(download_dataset_file):
        log_print(f"downloading dataset from {dataset_name} to {download_dataset_file}")
        data_list = download_dataset(dataset_name=dataset_name, dataset_subset=subset, num_samples=num_samples)
        save_json(data_list, download_dataset_file)
        log_print(f"saved downloaded dataset to {download_dataset_file}: {len(data_list)} items")
    else:
        with open(download_dataset_file, "r") as f:
            data_list = json.load(f)
        log_print(f"loading downloaded dataset from {download_dataset_file}: {len(data_list)} items")

    # 2. Search for PII using regex and starpii
    if not os.path.exists(search_dataset_file):
        log_print(f"searching for pii using regex and starpii from {download_dataset_file} to {search_dataset_file}")
        searched_dataset = search_pii(dataset=data_list, subset=subset)
        save_json(searched_dataset, search_dataset_file)
        log_print(f"saved searched dataset to {search_dataset_file}: {len(searched_dataset)} items")
    else:
        with open(search_dataset_file, "r") as f:
            searched_dataset = json.load(f)
        log_print(f"loading searched dataset from {search_dataset_file}: {len(searched_dataset)} items")
    
    # count the total number of pii records
    count = sum(len(item.get("piiRecords", [])) for item in searched_dataset)
    log_print(f"total pii records: {count}")

    # 3. Merge and clean the dataset
    if not os.path.exists(merge_dataset_file):
        log_print(f"merging and cleaning the dataset from {search_dataset_file} to {merge_dataset_file}")
        merged_cleaned_dataset = merge_clean_dataset(searched_dataset)
        save_json(merged_cleaned_dataset, merge_dataset_file)
        log_print(f"saved merged dataset to {merge_dataset_file}: {len(merged_cleaned_dataset)} items")
    else:
        with open(merge_dataset_file, "r") as f:
            merged_cleaned_dataset = json.load(f)
        log_print(f"loading merged dataset from {merge_dataset_file}: {len(merged_cleaned_dataset)} items")
    
    # get statistics of the dataset: count every value of detectedBy for each type of pii
    statistics = {}
    for item in merged_cleaned_dataset:
        for piiRecord in item.get("piiRecords", []):
            piiType = piiRecord.get("piiType")
            if piiType not in statistics:
                statistics[piiType] = {}
            if piiRecord.get("detectedBy") not in statistics[piiType]:
                statistics[piiType][piiRecord.get("detectedBy")] = 0
            statistics[piiType][piiRecord.get("detectedBy")] += 1
    log_print(f"statistics: {statistics}")


    # 4. Load the cache data
    if os.path.exists(cache_file):
        cache_data = load_json_safe(cache_file, default={})
        log_print(f"loading cache data from {cache_file} : {len(cache_data)} items (safe)")
    else:
        log_print(f"no cache data found at {cache_file}")
        cache_data = {}
        
    #in case of error, you can manually copy the backup file to the cache file
    if os.path.exists(cache_file_backup):
        cache_data_backup = load_json_safe(cache_file_backup, default={})
        if len(cache_data_backup) > len(cache_data):
            log_print(f"Using backup cache from {cache_file_backup} as it has more entries")
            cache_data = cache_data_backup
        elif len(cache_data_backup) < len(cache_data):
            save_json(cache_data, cache_file_backup)
            log_print(f"saved cache data to {cache_file_backup}")

    # 5. Refine the dataset
    if not os.path.exists(refine_dataset_file):
        log_print(f"refining the dataset from {merge_dataset_file} to {refine_dataset_file}")
        refined_dataset, cache_data = refine_dataset(
            dataset_pii=merged_cleaned_dataset,
            num_refine_samples=num_refine_samples,
            context_length=context_length,
            sample_threshold=sample_threshold,
            not_using_cache=not_using_cache,
            cache_file_path=cache_file
        )
        
        save_json(refined_dataset, refine_dataset_file)
        log_print(f"saved refined dataset to {refine_dataset_file}: {len(refined_dataset)} items")
    else:
        with open(refine_dataset_file, "r") as f:
            refined_dataset = json.load(f)
        log_print(f"loading refined dataset from {refine_dataset_file}: {len(refined_dataset)} items")
    
    # 6. Sample and save (if needed)
    # (Sampling and saving logic can be added here as needed.)
    train_dataset, val_dataset, test_dataset = sample_dataset(merged_cleaned_dataset, refined_dataset, n_train, n_val, n_test, visualize_size, visualize_dir)

    save_json(train_dataset, os.path.join(output_dir, "train_dataset.json"))
    save_json(val_dataset, os.path.join(output_dir, "val_dataset.json"))
    save_json(test_dataset, os.path.join(output_dir, "test_dataset.json"))

    log_print(f"saved train_dataset to {output_dir}/train_dataset.json, length: {len(train_dataset)}")
    log_print(f"saved val_dataset to {output_dir}/val_dataset.json, length: {len(val_dataset)}")
    log_print(f"saved test_dataset to {output_dir}/test_dataset.json, length: {len(test_dataset)}")

if __name__ == "__main__":
    main()
