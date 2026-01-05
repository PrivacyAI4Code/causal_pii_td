import json
import os
import re
import hashlib
import threading
import time
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from prompt import PROMPT_TEMPLATES, INSTRUCTIONS, quick_check
from utils import log_print, save_json, load_json_safe


def parse_content2json(content: str):
    """
    Attempts to parse the LLM output as JSON, or extract score/reason if not valid JSON.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting fenced JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: extract score and reason loosely
    try:
        score = int(re.search(r'"score":\s*(\d+)', content).group(1))
        reason = re.search(r'"reason":\s*"([^"]*)"', content).group(1)
        return {"score": score, "reason": reason}
    except Exception:
        return {"score": 0, "reason": content}


def refine_value(client, prompt, model, temperature, instructions):
    """
    Refine the value based on its PII type using the LLM.
    """
    # prompt = PROMPT_TEMPLATES[piiType]["user"].format(context=context, value=value)
    # model = "gpt-5-mini"
    # temperature = 0.2
    # instructions = INSTRUCTIONS[piiType]
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        # temperature=temperature,
    )
    content = response.output_text
    result = parse_content2json(content)
    # Defensive: ensure keys exist
    score = result.get("score", 0)
    reason = result.get("reason", "")
    # return value, score, reason
    return score, reason


def refine_dataset(
    dataset_pii,
    num_refine_samples,
    context_length,
    sample_threshold,
    not_using_cache=[],
    target_pii_types=["email", "password", "name", "username", "ip_address", "key"],
    max_workers: int = 16,
    cache_file_path: str = None,
):
    """
    Refine a dataset of PII records using multi-threading per PII type. Preserves the
    original semantics: processes types in fixed order, skips same `id` once a high
    score is found, and stops early per type when enough high-sensitive samples are found.
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Organize dataset by piiType
        dataset_by_piiType = defaultdict(list)
        random.seed(42)
        random.shuffle(dataset_pii)
        for idx, pii in enumerate(dataset_pii):
            for piiRecord in pii.get("piiRecords", []):
                piiRecord["id"] = idx
                dataset_by_piiType[piiRecord["piiType"]].append(piiRecord)
        log_print(f"dataset_by_piiType: {list(dataset_by_piiType.keys())}")
        for piiType, dataset in dataset_by_piiType.items():
            log_print(f"piiType: {piiType}, {len(dataset)} items")

        refined_dataset = []
        used_ids = set()
        used_ids_lock = threading.Lock()
        cache_lock = threading.Lock()
        process_info = []

        order_types = ["ip_address", "key", "password", "email", "name", "username"]
        if os.path.exists(cache_file_path):
            cache_data = load_json_safe(cache_file_path, default={})
        else:
            cache_data = {}

        for piiType in order_types:
            dataset = dataset_by_piiType[piiType]
            if piiType not in target_pii_types:
                continue
            if piiType == "key":
                dataset = sorted(dataset, key=lambda x: x.get("detectedBy") != "regex")

            high_sensitive_threshold = sample_threshold[piiType]
            context_length_current_type = context_length[piiType]

            # Counters per-type
            count_all_piis = 0
            count_skip_quick_check = 0
            count_miss_location = 0
            count_skip_same_id = 0
            count_llm_judged = 0
            count_cache_hits = 0
            count_above_threshold = 0
            count_returned_piis = 0

            counters_lock = threading.Lock()
            stop_event = threading.Event()
            dataset_refined = []

            def worker(record):
                nonlocal dataset_pii, piiType, high_sensitive_threshold
                nonlocal context_length_current_type, cache_data

                if stop_event.is_set():
                    return {"dropped": True}

                idx_local = record["id"]
                record_local = record
                content_local = dataset_pii[idx_local]["content"]
                piivalue = record_local["value"]
                location_start = record_local["location_start"]
                location_end = record_local["location_end"]
                detectedBy = record_local.get("detectedBy", "")

                # Validate type and location
                if piiType != record_local["piiType"] or piivalue != content_local[location_start: location_end]:
                    return {"miss_location": 1}

                # quick_check
                value_checked, loc_start_checked, loc_end_checked, explanation_q = quick_check(
                    piivalue, location_start, location_end, piiType, content_local
                )
                if value_checked is None:
                    return {"skip_quick_check": 1}

                # Updated value and location after quick_check
                value = value_checked
                location_start = loc_start_checked
                location_end = loc_end_checked

                # Re-check same-id skip as used_ids can change concurrently
                with used_ids_lock:
                    if idx_local in used_ids:
                        return {"skip_same_id": 1}

                # Build context and cache key
                context = content_local[max(0, location_start - context_length_current_type): min(len(content_local), location_end + context_length_current_type)]
                prompt = PROMPT_TEMPLATES[piiType]["user"].format(context=context, value=value)
                model = "gpt-5-mini"
                temperature = None
                instructions = INSTRUCTIONS[piiType]

                cache_key_raw = f"{model}:{instructions}:{prompt}:{temperature}"
                cache_key = hashlib.md5(cache_key_raw.encode('utf-8')).hexdigest()

                cached_tuple = None
                if piiType not in not_using_cache:
                    with cache_lock:
                        cached_tuple = cache_data.get(cache_key)

                judged_by_llm = 0
                cache_hit = 0
                if cached_tuple is not None and piiType not in not_using_cache:
                    score_out, explanation_out = cached_tuple
                    cache_hit = 1
                else:
                    if stop_event.is_set():
                        return {"dropped": True}
                    score_out, explanation_out = refine_value(client, prompt, model, temperature, instructions)
                    judged_by_llm = 1
                    with cache_lock:
                        cache_data[cache_key] = (score_out, explanation_out)

                record_judged = {
                    "id": idx_local,
                    "value": value,
                    "location_start": location_start,
                    "location_end": location_end,
                    "piiType": piiType,
                    "detectedBy": detectedBy,
                    "isHumanReviewed": False,
                    "score": score_out,
                    "reason": explanation_out,
                }

                return {
                    "judged": judged_by_llm,
                    "cache_hit": cache_hit,
                    "record": record_judged,
                    "above": score_out >= high_sensitive_threshold,
                    "idx": idx_local,
                    "score": score_out,
                    "value": value,
                }

            def process_result(result):
                nonlocal count_skip_quick_check, count_miss_location, count_skip_same_id
                nonlocal count_llm_judged, count_cache_hits, count_above_threshold
                nonlocal count_returned_piis, dataset_refined

                if result is None:
                    return
                if result.get("dropped"):
                    return
                if result.get("miss_location"):
                    with counters_lock:
                        count_miss_location += 1
                    return
                if result.get("skip_quick_check"):
                    with counters_lock:
                        count_skip_quick_check += 1
                    return
                if result.get("skip_same_id"):
                    with counters_lock:
                        count_skip_same_id += 1
                    return

                with counters_lock:
                    count_llm_judged += result.get("judged", 0)
                    count_cache_hits += result.get("cache_hit", 0)

                if stop_event.is_set():
                    return

                record_j = result["record"]
                above = result["above"]

                dataset_refined.append(record_j)
                with counters_lock:
                    count_returned_piis += 1

                if above:
                    with used_ids_lock:
                        used_ids.add(result["idx"])
                    with counters_lock:
                        count_above_threshold += 1
                    log_print(f"type: {piiType}, score: {result['score']}, current count_above_threshold: {count_above_threshold}, value: {result['value']}")
                    if count_above_threshold >= num_refine_samples:
                        stop_event.set()

            # Schedule tasks with bounded inflight to improve early stopping efficacy
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                inflight = set()
                for record in dataset:
                    with counters_lock:
                        count_all_piis += 1

                    # Pre-check same id to avoid unnecessary work
                    with used_ids_lock:
                        if record["id"] in used_ids:
                            with counters_lock:
                                count_skip_same_id += 1
                            continue

                    if stop_event.is_set():
                        break

                    fut = executor.submit(worker, record)
                    inflight.add(fut)

                    if len(inflight) >= max_workers:
                        done, _ = next(((d, None) for d in as_completed(inflight, timeout=None)), (None, None))
                        if done is not None:
                            inflight.remove(done)
                            try:
                                res = done.result()
                            except Exception as e:
                                log_print(f"Exception in worker: {e}")
                                res = None
                            process_result(res)

                # Drain remaining tasks
                for done in as_completed(inflight):
                    try:
                        res = done.result()
                    except Exception as e:
                        log_print(f"Exception in worker: {e}")
                        res = None
                    process_result(res)

            refined_dataset.extend(dataset_refined)
            info = f"for {piiType}, there are {count_all_piis} records, {count_skip_quick_check} records are skipped due to quick check, {count_miss_location} records are skipped due to location mismatch, {count_skip_same_id} records are skipped due to same id, {count_llm_judged} records are judged by llm, {count_cache_hits} records are hit in cache, {count_above_threshold} records are above threshold, {count_returned_piis} records are returned"
            log_print(info)
            process_info.append(info)
        
        for info in process_info:
            log_print(info)
        
        save_json(cache_data, cache_file_path)

        return refined_dataset, cache_data
    except Exception as e:
        # save cache_data to cache_file_path
        save_json(cache_data, cache_file_path)
        raise e
    except KeyboardInterrupt:
        # save cache_data to cache_file_path
        save_json(cache_data, cache_file_path)
        raise KeyboardInterrupt