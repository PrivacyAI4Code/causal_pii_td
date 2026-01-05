import os
import re
import json
import csv
from typing import List, Dict, Any, Optional
import pickle
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import datetime
import inspect

def flatten_config(config: dict) -> dict:
    flat_dict = {}
    def recurse(d):
        for k, v in d.items():
            if isinstance(v, dict):
                recurse(v)
            else:
                flat_dict[k] = v
    recurse(config)
    return flat_dict

def sort_list_by_last_number(input_list: List[Any]) -> List[Any]:
    """Sort a list of items based on the last number found in each item's string representation.
    
    Args:
        input_list: List of items to be sorted
        
    Returns:
        List sorted by the last number found in each item's string representation
    """
    def extract_last_number(item):
        # Find all numbers in the string, return the last one as int, or 0 if none found
        numbers = re.findall(r'\d+', str(item))
        return int(numbers[-1]) if numbers else 0

    # Sort by the last number, ascending from 1 to 100
    return sorted(input_list, key=extract_last_number)

def find_pkl_files(directory: str, sort: bool = False) -> List[str]:
    """Find all .pkl files in the given directory and its subdirectories.
    
    Args:
        directory: Root directory to search for .pkl files
        
    Returns:
        List of paths to .pkl files, sorted by the last number in their filenames
    """
    pkl_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    if sort:
        pkl_files = sort_list_by_last_number(pkl_files)
    return pkl_files

def find_jsonl_files(directory: str, sort: bool = False) -> List[str]:
    """Find all .jsonl files in the given directory and its subdirectories.

    Args:
        directory: Root directory to search for .jsonl files
        
        Returns:
        List of paths to .jsonl files, sorted by the last number in their filenames
    """
    jsonl_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    if sort:
        jsonl_files = sort_list_by_last_number(jsonl_files)
    return jsonl_files


def load_file(file_path: str) -> List[Any]:
    """Load data from a file.
    
    Args:
        file_path (str): Path to the file containing the data
        
    Returns:    
        List[Any]: List of items from the data file
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file type is not supported (.pkl, .json, .txt, or .csv)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return f.readlines()
    elif file_path.endswith('.csv'):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            return list(reader)
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
   
# Mapping from StarPII model labels to our PII types
map_starpii_to_pii_types = {
    "NAME": "name",
    "EMAIL": "email",
    "IP_ADDRESS": "ip_address",
    "KEY": "key",
    "PASSWORD": "password",
    "USERNAME": "username",
}

def chunk_text(text, tokenizer, max_length=1024, stride=512):
    """
    Split text into overlapping chunks for processing by the NER model.
    Returns list of (chunk_text, start_position) tuples.
    
    Optimized version using offset_mapping for efficient position calculation.
    """
    # Use offset_mapping to get character positions directly
    encoded = tokenizer(
        text, 
        return_tensors="pt", 
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    
    tokens = encoded["input_ids"][0]
    offset_mapping = encoded["offset_mapping"][0]
    
    chunks = []
    for i in range(0, len(tokens), stride):
        # Get token slice - be more conservative to avoid retokenization issues
        actual_max = max_length - 10  # More conservative buffer for token boundary issues
        end_idx = min(i + actual_max, len(tokens))
        
        # Get character positions from offset mapping
        start_char = offset_mapping[i][0].item() if i < len(offset_mapping) else 0
        end_char = offset_mapping[end_idx-1][1].item() if end_idx > 0 else len(text)
        
        # Extract chunk text directly from original text
        chunk_text = text[start_char:end_char]
        chunks.append((chunk_text, start_char))
        
        # Break if we've reached the end
        if end_idx >= len(tokens):
            break
    
    return chunks

# def chunk_text_batch(texts, tokenizer, max_length=1024, stride=512):
#     """
#     Batch version of chunk_text for multiple texts.
#     Returns list of (text_idx, chunk_text, start_position) tuples.
#     """
#     all_chunks = []
    
#     for text_idx, text in enumerate(texts):
#         chunks = chunk_text(text, tokenizer, max_length, stride)
#         for chunk_text, start_pos in chunks:
#             all_chunks.append((text_idx, chunk_text, start_pos))
    
#     return all_chunks

def get_pipeline_ner():
    # Initialize StarPII model for NER
    import torch
    model_name = "bigcode/starpii"
    
    # Determine device
    device = 0 if torch.cuda.is_available() else -1
    print(f"🚀 Using device: {'GPU' if device == 0 else 'CPU'}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    max_length = model.config.max_position_embeddings
    
    # Create pipeline with device specification
    nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, 
                           aggregation_strategy="simple", device=device)

    return nlp_pipeline, tokenizer, max_length


def process_ner_batch(texts, nlp_pipeline, batch_size=512, max_length=1024):
    """
    Process multiple texts through NER pipeline in batches for better GPU utilization.
    
    Args:
        texts: List of text strings to process
        nlp_pipeline: The NER pipeline
        batch_size: Number of texts to process in each batch
        max_length: Maximum token length for each text
    
    Returns:
        List of NER results corresponding to each input text
    """
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            # Process batch
            batch_results = nlp_pipeline(batch_texts)
            # print(f"batch_size: {batch_size}, len(batch_texts): {len(batch_texts)}, len(batch_results): {len(batch_results)}")
            # exit(1)
            
            # Ensure batch_results is a list of lists
            if len(batch_texts) == 1:
                batch_results = [batch_results]
            
            all_results.extend(batch_results)
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            pass
    
    return all_results


def extract_pii_from_chunks_batch(content_list, nlp_pipeline, tokenizer, max_length, stride=None, batch_size=4096, debug=False):
    """
    Optimized batch processing of PII detection using StarPII NER model.
    
    Args:
        content_list: List of text contents to process
        nlp_pipeline: The NER pipeline
        tokenizer: The tokenizer
        max_length: Maximum token length for chunks
        stride: Stride for overlapping chunks (default: max_length//2)
        batch_size: Batch size for NER processing
        debug: Enable debug output
    
    Returns:
        List of PII results for each content
    """
    if stride is None:
        stride = max_length // 2
    
    # print(f"📦 Processing {len(content_list)} contents with batch_size={batch_size}")
    # exit(1)
    
    # Step 1: Create all chunks for all contents
    all_chunks = []
    chunk_to_content = []  # Maps chunk index to (content_index, offset)
    
    for content_idx, content in enumerate(content_list):
        chunks = chunk_text(content, tokenizer, max_length, stride)
        if debug:
            print(f"      🔧 Content {content_idx}: created {len(chunks)} chunks")
        for text_chunk, offset in chunks:
            all_chunks.append(text_chunk)
            chunk_to_content.append((content_idx, offset))
    
    if debug:
        print(f"      🔧 Total chunks to process: {len(all_chunks)}")
    
    # Step 2: Process all chunks in batches
    if all_chunks:
        if debug:
            print(f"      🔧 Calling process_ner_batch...")
        chunk_results = process_ner_batch(all_chunks, nlp_pipeline, batch_size)
        if debug:
            print(f"      🔧 Got {len(chunk_results)} chunk results")
    else:
        chunk_results = []
    
    # Step 3: Group results back by original content
    content_results = [[] for _ in content_list]
    
    for chunk_idx, chunk_result in enumerate(chunk_results):
        if chunk_idx >= len(chunk_to_content):
            continue
            
        content_idx, offset = chunk_to_content[chunk_idx]
        
        if chunk_result:
            for result in chunk_result:
                # Ensure result is a dictionary with required keys
                if not isinstance(result, dict) or 'end' not in result:
                    if debug:
                        print(f"      🔧 Skipping invalid result: {result}")
                    continue
                
                # Skip entities that end at text boundary (likely incomplete)
                if result['end'] == len(all_chunks[chunk_idx]):
                    continue
                
                # Adjust positions to original content coordinates
                adjusted_result = {
                    'entity_group': result['entity_group'],
                    'start': result['start'] + offset,
                    'end': result['end'] + offset,
                    'word': result['word'],
                    'score': result['score']
                }
                content_results[content_idx].append(adjusted_result)
    
    if debug:
        print(f"      🔧 Final results: {[len(r) for r in content_results]}")
    
    return content_results


# save json (atomic write)
def save_json(data, file_path, indent=4):
    if not file_path.endswith(".json"):
        file_path = file_path + ".json"
    # Write atomically to prevent partial/corrupted files on crash or concurrent writes
    import os
    import tempfile
    dir_path = os.path.dirname(file_path) or "."
    os.makedirs(dir_path, exist_ok=True)
    tmp_prefix = os.path.basename(file_path) + ".tmp."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_path, prefix=tmp_prefix) as tmp:
        json.dump(data, tmp, indent=indent)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, file_path)


def load_json_safe(file_path, default=None, try_repair=True):
    """
    Safely load JSON. If decoding fails, optionally attempt a simple repair by truncating
    to the last closing brace, otherwise return `default`.
    """
    import os
    if default is None:
        default = {}
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e1:
        # Best-effort repair for truncated JSON objects
        if try_repair and os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    s = f.read()
                last_brace = s.rfind("}")
                if last_brace != -1:
                    repaired = s[: last_brace + 1]
                    try:
                        obj = json.loads(repaired)
                        # Persist the repaired content atomically
                        save_json(obj, file_path)
                        return obj
                    except Exception:
                        pass
            except Exception:
                pass
        # Fall back to default if repair failed
        return default

# save jsonl
def save_jsonl(data, file_path):
    if not file_path.endswith(".jsonl"):
        file_path = file_path + ".jsonl"
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def log_print(*args, **kwargs):
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    now = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    print(f"{now}-{info.filename}-{info.function}-{info.lineno}:  {msg}", **kwargs)



