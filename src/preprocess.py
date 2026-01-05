"""
Dataset Merger

This script merges two datasets: PII detection results and tools-based detection results.
It processes matching files and combines their PII records while avoiding duplicates.

The script processes files in batches and saves the merged results.
"""
import os
import json
import argparse
import logging
import sys
from datetime import datetime
from typing import List, Dict, Tuple

from utils import log_print
from pii_data import CodeFile_KEYS, pii_types_list
from pii_regex import ip_has_digit, matches_date_pattern, not_ip_address

from collections import defaultdict

def validate_key(string: str) -> str:
    """
    Check if the key is valid
    """
    if string.lower() == "bigcode_key":
        return "key"
    else:
        if string in pii_types_list:
            return string
        else:
            raise ValueError(f"Invalid key: {string}")

def validate_location(content: str, location_start: int, location_end: int, value: str) -> Tuple[int, int]:
    """
    Check if the value is at the right position using a neighborhood-first strategy.
    """
    if value == "":
        return -1, -1
    # Fast path: exact match at provided coordinates
    if value and content[location_start:location_end] == value:
        return location_start, location_end
    else:
        window = 20
        orig_start = location_start
        orig_end = location_end
        base_start = max(0, orig_start - window)
        base_end = min(len(content), orig_end + window)
        new_start = content.find(value, base_start, base_end)
        if new_start != -1:
            return new_start, new_start+len(value)
        else:
            window = 100
            base_start = max(0, orig_start - window)
            base_end = min(len(content), orig_end + window)
            new_start = content.find(value, base_start, base_end)
            if new_start != -1:
                return new_start, new_start+len(value)
            else:
                return -1, -1

def merge_pii_records(content: str, pii_records_1: List[Dict], pii_records_2: List[Dict], 
                     is_validate_key: bool = True, is_validate_location: bool = True) -> List[Dict]:
    """
    Merge the two piiRecords, avoiding duplicates based on location and value
    """
    existing_records = set()
    pii_records_merged = []
    for record in pii_records_1 + pii_records_2:
        record_key = (record["location_start"], record["location_end"], record["value"])
        if record_key not in existing_records:
            existing_records.add(record_key)
            pii_records_merged.append(record)

    pii_records_merged_checked = []
    for record in pii_records_merged:
        try:
            if is_validate_key:
                record["piiType"] = validate_key(record["piiType"])
            if is_validate_location:
                record["location_start"], record["location_end"] = validate_location(
                    content, record["location_start"], record["location_end"], record["value"])
            record["piiId"] = len(pii_records_merged_checked)
            pii_records_merged_checked.append(record)
        except ValueError as e:
            log_print(f"Error: {e}")
            continue

    return pii_records_merged_checked

def merge_data(data_1: Dict, data_2: Dict) -> Dict:
    """
    Merge the two datasets
    """
    if not (data_1.keys() == data_2.keys() == CodeFile_KEYS):
        raise ValueError("Input datasets must have the same keys as CodeFile_KEYS")
    data_merged = data_1.copy()
    data_merged["piiRecords"] = merge_pii_records(data_1["content"], data_1["piiRecords"], data_2["piiRecords"])
    return data_merged

def process_datasets(dataset_pii: List[Dict], dataset_tools: List[Dict], 
                    is_contain_pii: bool = True, flag_test: bool = False) -> List[Dict]:
    """
    Process and merge two datasets while maintaining fileId continuity
    """
    dataset_merged = []
    pointer_pii = pointer_tools = 0
    while pointer_pii < len(dataset_pii) and pointer_tools < len(dataset_tools):
        id_pii = dataset_pii[pointer_pii]["fileId"]
        id_tools = dataset_tools[pointer_tools]["fileId"]
        if id_pii == id_tools:
            dataset_merged.append(merge_data(dataset_pii[pointer_pii], dataset_tools[pointer_tools]))
            pointer_pii += 1
            pointer_tools += 1
        elif id_pii < id_tools:
            dataset_merged.append(dataset_pii[pointer_pii])
            pointer_pii += 1
        else:
            dataset_merged.append(dataset_tools[pointer_tools])
            pointer_tools += 1
    while pointer_pii < len(dataset_pii):
        dataset_merged.append(dataset_pii[pointer_pii])
        pointer_pii += 1
    while pointer_tools < len(dataset_tools):
        dataset_merged.append(dataset_tools[pointer_tools])
        pointer_tools += 1
    
    if flag_test and len(dataset_merged) > 10:
        return dataset_merged[:10]
    
    # Filter to only contain PII if requested
    if is_contain_pii:
        dataset_merged_only_contain_pii = []
        for item in dataset_merged:
            if len(item["piiRecords"]) > 0:
                dataset_merged_only_contain_pii.append(item)
        return dataset_merged_only_contain_pii
    else:
        return dataset_merged


def merge_file(dataset_list: List[List[Dict]]) -> Tuple[List[Dict], int, int]:
    """
    Merge datasets and remove duplicates, returning the merged dataset and size statistics
    """
    before_size = 0
    after_size = 0
    
    # Create a copy to avoid modifying the original
    dataset_merged = dataset_list[0].copy()
    if len(dataset_list) > 1:
        raise ValueError("Only one dataset is supported")
    
    for idx, data_sample in enumerate(dataset_merged):
        before_size += len(data_sample["piiRecords"])
        existing_records = set()
        pii_records_merged = []
        
        for record in data_sample["piiRecords"]:
            # remove space that is in the front or back of the value, if removes any space, update the location_start and location_end
            # record["value"] = record["value"].strip()
            stripped = record["value"].strip()
            offset = record["value"].find(stripped)
            record["location_start"] += offset
            record["location_end"] = record["location_start"] + len(stripped)
            record["value"] = stripped

            record_key = (record["location_start"], record["location_end"], record["value"])
            if record_key not in existing_records:
                existing_records.add(record_key)
                pii_records_merged.append(record)
        
        data_sample["piiRecords"] = pii_records_merged
        
        try:
            # validate the location
            for record in data_sample["piiRecords"]:
                record["location_start"], record["location_end"] = validate_location(
                    data_sample["content"], record["location_start"], record["location_end"], record["value"])
            after_size += len(data_sample["piiRecords"])
        except Exception as e:
            logging.error(f"Error processing file {idx} of {len(dataset_merged)}: {e}")
            continue
    
    return dataset_merged, before_size, after_size

def clean_dataset(dataset_input: List[Dict]) -> List[Dict]:
    """
    Clean the dataset
    1.notes from https://huggingface.co/bigcode/starpii:
    - Ignore secrets with less than 4 characters.
    - Detect full names only.
    - Ignore detected keys with less than 9 characters.
    - Ignore IP addresses that aren't valid or are private (non-internet facing) using the ipaddress python package. We also ignore IP addresses from popular DNS servers. We use the same list as in this paper.
    """
    num_pii_records_before = 0
    num_pii_records_after = 0
    dataset_output = []
    for record in dataset_input:
        pii_records = record["piiRecords"]
        num_pii_records_before += len(pii_records)
        pii_records_clean = []
        for pii_record in pii_records:
            # Ignore secrets with less than 4 characters.
            if len(pii_record["value"]) < 4:
                continue
            if len(pii_record["value"]) > 300: # avoid too long pii
                continue
            if pii_record["piiType"] == "name":
                # full name likely have space
                if " " not in pii_record["value"]:
                    continue
            if pii_record["piiType"] == "key":
                if len(pii_record["value"]) < 9:
                    continue
            if pii_record["piiType"] == "ip_address":
                value = pii_record["value"]
                if not ip_has_digit(value) :
                    continue
                if not_ip_address(value):
                    continue
                if matches_date_pattern(value):
                    continue
            pii_records_clean.append(pii_record)
        record["piiRecords"] = pii_records_clean
        num_pii_records_after += len(pii_records_clean)
        dataset_output.append(record)
    log_print(f"clean: {len(dataset_output)} items with {num_pii_records_after} PII items")
    if num_pii_records_before > 0:
        log_print(f"reduction: {((num_pii_records_before - num_pii_records_after) / num_pii_records_before * 100):.2f}%")
    else:
        log_print("reduction: 0.00% (no PII records before cleaning)")

    return dataset_output
            
def split_dataset_by_piiType(dataset: List[Dict]) -> Dict[str, List[Dict]]:
    """
    split the dataset by piiType
    """
    dataset_by_piiType = defaultdict(list)
    
    for idx, record in enumerate(dataset):
        for piiRecord in record["piiRecords"]:
            piiType = piiRecord["piiType"]
            record_copy = {
                "id": idx,
                "piiRecord": piiRecord
            }
            dataset_by_piiType[piiType].append(record_copy)
            
    return dict(dataset_by_piiType)

def merge_clean_dataset(dataset_pii: List[Dict]):
    """
    Merge and clean the dataset
    """
    dataset_list = [dataset_pii]
    dataset_merged, before_size, after_size = merge_file(dataset_list)
    log_print(f"merge: {len(dataset_merged)} items with {after_size} PII items from {before_size} PII items")
    
    dataset_cleaned = clean_dataset(dataset_merged)
    clean_count = 0
    for record in dataset_cleaned:
        clean_count += len(record["piiRecords"])
    log_print(f"after clean: {len(dataset_cleaned)} items with {clean_count} PII items")

    # # sotre all pii records
    # dataset_pii = []
    # for idx, record in enumerate(dataset_cleaned):
    #     for piiRecord in record["piiRecords"]:
    #         piiType = piiRecord["piiType"]
    #         record_copy = {"id": idx,}
    #         for k, v in piiRecord.items():
    #             record_copy[k] = v
    #         dataset_pii.append(record_copy)

    return dataset_cleaned
        
