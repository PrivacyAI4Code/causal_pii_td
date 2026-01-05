"""
PII (Personally Identifiable Information) 

This script scans code files for various types of PII using multiple detection methods:
1. NER (Named Entity Recognition) using StarPII model
2. Regex patterns for specific PII types

The script processes code files in batches and saves the results with detected PII information.
"""
import os
import json
import argparse
import datetime
from tqdm import tqdm

from pii_regex import find_pii_by_regex
from pii_data import CodeFile, PiiRecord
from utils import chunk_text, map_starpii_to_pii_types, load_file, extract_pii_from_chunks_batch, get_pipeline_ner, log_print
# from search_tools import detect_pii_with_tools

# PII types will be searched with regex
regex_match = [
    "email",
    "ip_address", 
    # "gender",
    "key",
    # "credit_card",
]

_regex_cache = {}

def process_file_with_batch(dataset, nlp_pipeline, tokenizer, max_length, suffix, batch_size=256):
    """Process dataset using optimized batch processing for NER."""
    log_print("Using batch processing mode...")

    # Collect content and build index mapping
    content_list = []
    idx_to_content_idx = {}  # Maps dataset idx to content_list idx

    for idx, data in enumerate(dataset):
        content = data.get("content", "")
        if content:
            content_idx = len(content_list)
            content_list.append(content)
            idx_to_content_idx[idx] = content_idx

    log_print(f"Extracted {len(content_list)} valid contents for batch processing")

    # Batch process NER detection
    ner_start_time = datetime.datetime.now()
    try:
        batch_ner_results = extract_pii_from_chunks_batch(
            content_list, nlp_pipeline, tokenizer, max_length,
            batch_size=batch_size, debug=False
        )
        log_print(f"Batch NER completed in {(datetime.datetime.now() - ner_start_time).total_seconds():.2f}s")
    except Exception as e:
        log_print(f"Error in batch NER processing: {e}")
        batch_ner_results = [[] for _ in content_list]

    # Process each record with all detection methods
    data_list_pii = []
    count = 0

    with tqdm(total=len(dataset), desc="Processing records") as pbar:
        for idx, data in enumerate(dataset):
            # Update progress bar every 1000 records
            if idx % 1000 == 0 and idx > 0:
                pbar.update(1000)

            content = data.get("content", "")
            pii_results = []

            # 1. Regex detection
            for pii_type in regex_match:
                try:
                    regex_results = find_pii_by_regex(content, pii_type)
                    if regex_results:
                        for result in regex_results:
                            result['piiId'] = len(pii_results)
                            pii_results.append(PiiRecord(**result))
                except Exception as e:
                    log_print(f"Regex error for {pii_type}: {e}")
                    continue

            # 2. Add batch-processed NER results
            content_idx = idx_to_content_idx.get(idx)
            if content_idx is not None and content_idx < len(batch_ner_results):
                for result in batch_ner_results[content_idx]:
                    try:
                        pii_results.append(PiiRecord(
                            piiId=len(pii_results),
                            piiType=map_starpii_to_pii_types.get(result['entity_group'], result['entity_group']),
                            location_start=result['start'],
                            location_end=result['end'],
                            value=result['word'],
                            detectedBy="StarPII",
                            timestamp=datetime.datetime.now().isoformat(),
                            notes=f"confidence: {result['score']:.2f}",
                            isHumanReviewed=False,
                            confidenceScore=float(result.get('score', -1.0)),
                        ))
                    except Exception as e:
                        log_print(f"NER result processing error: {e}")
                        continue

            # 3. Tool detection (currently commented out)
            # try:
            #     tool_results = detect_pii_with_tools(content, suffix=suffix, tools_list=[])#"presidio" "trufflehog"
            #     for result in tool_results:
            #         result['piiId'] = len(pii_results)
            #         pii_results.append(PiiRecord(**result))
            # except Exception as e:
            #     log_print(f"Tool detection error: {e}")

            data["piiRecords"] = pii_results
            # Only keep records with PII
            if len(pii_results) > 0:
                data_list_pii.append(data)
                count += len(pii_results)

        # Final progress bar update for any remaining records
        remaining = len(dataset) % 1000
        if remaining > 0:
            pbar.update(remaining)

    return data_list_pii


# def process_file_individual(dataset, nlp_pipeline, tokenizer, max_length, suffix):
#     """Process dataset using individual processing for each record."""
#     log_print("Using individual processing mode...")
    
#     data_list_pii = []
#     count = 0
    
#     with tqdm(total=len(dataset), desc="Processing records") as pbar:
#         for idx, data in enumerate(dataset):
#             if idx % 1000 == 0 and idx > 0:
#                 pbar.update(1000)
            
#             content = data.get("content", "")
#             pii_results = []

#             # 1. Regex detection
#             for pii_type in regex_match:
#                 try:
#                     regex_results = find_pii_by_regex(content, pii_type)
#                     if regex_results:
#                         for result in regex_results:
#                             result['piiId'] = len(pii_results)
#                             pii_results.append(PiiRecord(**result))
#                 except Exception as e:
#                     continue
            
#             # 2. Individual NER detection
#             try:
#                 chunks = chunk_text(content, tokenizer, max_length=max_length, stride=max_length//2)  
#                 for text, offset in chunks:
#                     ner_results = nlp_pipeline(text)
#                     if ner_results:
#                         for result in ner_results:
#                             if result['end'] == len(text):
#                                 continue
#                             pii_results.append(PiiRecord(
#                                 piiId=len(pii_results),
#                                 piiType=map_starpii_to_pii_types.get(result['entity_group'], result['entity_group']),
#                                 location_start=result['start'] + offset,
#                                 location_end=result['end'] + offset,
#                                 value=result['word'],
#                                 detectedBy="StarPII",
#                                 timestamp=datetime.datetime.now().isoformat(),
#                                 notes=f"confidence: {result['score']:.2f}",
#                                 isHumanReviewed=False,
#                                 confidenceScore=float(result.get('score', -1.0)),
#                             ))
#             except Exception as e:
#                 pass
            
#             # 3. Tool detection
#             try:
#                 tool_results = detect_pii_with_tools(content, suffix=suffix, tools_list=["trufflehog"]) #, "trufflehog" "presidio"
#                 for result in tool_results:
#                     result['piiId'] = len(pii_results)
#                     pii_results.append(PiiRecord(**result))
#             except Exception as e:
#                 pass

#             data["piiRecords"] = pii_results
#             # 只保留包含PII的记录
#             if len(pii_results) > 0:
#                 data_list_pii.append(data)
#                 count += len(pii_results)
            
#             log_print(data)
#             log_print(data["piiRecords"])
#             exit(1)
    
#         # 正确的最终进度更新
#         remaining = len(dataset) % 1000
#         if remaining > 0:
#             pbar.update(remaining)
    
#     return data_list_pii, count


def process_file(file_path: str, nlp_pipeline, tokenizer, max_length: int, suffix: str, batch_size: int = 4096, use_batch: bool = True):
    """
    Process a single file and return the dataset with PII records.
    Args:
        use_batch: If True, use batch processing; if False, use individual processing
    """
    try:
        dataset = load_file(file_path)
        log_print(f"Processing {file_path} with {len(dataset)} records")
    except Exception as e:
        log_print(f"Error loading file {file_path}: {e}")
        return None, 0
    
    if use_batch:
        return process_file_with_batch(dataset, nlp_pipeline, tokenizer, max_length, suffix, batch_size)
    else:
        # return process_file_individual(dataset, nlp_pipeline, tokenizer, max_length, suffix)
        raise Exception("Individual processing is not supported")


def search_pii(dataset, subset, batch_size=512):
    """
    Search PII in the input file and save the results to the output directory.
    """
    if subset.lower() == "java":
        suffix = ".java"
    elif subset.lower() == "python":
        suffix = ".py"
    else:
        raise Exception(f"Unsupported subset: {subset}")

    # 1. get nlp_pipeline (starpii)
    nlp_pipeline, tokenizer, max_length = get_pipeline_ner()

    # 2. process files
    log_print("processing files with batch mode")
    dataset_pii = process_file_with_batch(dataset, nlp_pipeline, tokenizer, max_length, suffix, batch_size)

    return dataset_pii
