import os
import threading
from typing import Optional
from datasets import load_dataset
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from smart_open import open as smart_open

from utils import log_print

# --- Global Variables & Locks ---
lock = threading.Lock()
stop_processing_flag = threading.Event()  # Used to signal all threads to stop


def get_s3_client():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables must be set")
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    s3_client = session.client("s3")
    return s3_client


def process_file(file_item: dict, s3_boto_client: boto3.client) -> Optional[dict]:
    """
    Downloads and decodes a single file from S3.
    Args:
        file_item: A dictionary containing metadata about the file, including 'blob_id' and 'src_encoding'.
        s3_boto_client: An initialized Boto3 S3 client.
    Returns:
        A dictionary with file content and metadata, or None if an error occurs.
    """
    s3_url = f"s3://softwareheritage/content/{file_item['blob_id']}"
    try:
        # smart_open uses boto3 client under the hood if transport_params are set.
        # Ensure your s3_boto_client is configured for the desired region if necessary.
        # Try with compression first, fall back to no compression if it fails
        try:
            with smart_open(s3_url, "rb", compression=".gz", transport_params={"client": s3_boto_client}) as fin:
                content_bytes = fin.read()
        except Exception:
            # If compression fails, try without compression
            with smart_open(s3_url, "rb", transport_params={"client": s3_boto_client}) as fin:
                content_bytes = fin.read()
                
        content = content_bytes.decode(file_item["src_encoding"])
        return {
            "blobId": file_item["blob_id"],
            "directoryId": file_item.get("directory_id"), # Use .get for robustness
            "path": file_item.get("path"),
            "contentId": file_item.get("content_id"),
            "detectedLicenses": file_item.get("detected_licenses"),
            "licenseType": file_item.get("license_type"),
            "repoName": file_item.get("repo_name"),
            "githubId": file_item.get("github_id"),
            "language": file_item.get("language"),
            "content": content,
            "lengthBytes": file_item.get("length_bytes", len(content_bytes)), # Fallback for length
            "extension": file_item.get("extension"),
        }
    except UnicodeDecodeError as ude:
        log_print(f"Encoding error for {s3_url} with encoding {file_item.get('src_encoding', 'N/A')}: {ude}")
        return None
    except boto3.exceptions.Boto3Error as s3e: # Catch Boto3 specific errors
        log_print(f"S3 related error for {s3_url}: {s3e}")
        return None
    except Exception as e:
        # Log other unexpected errors
        log_print(f"Error reading or processing {s3_url}: {type(e).__name__} - {e}")
        return None


def download_dataset(dataset_name, dataset_subset, num_samples, max_workers=8):
    """
    Download and process files from a streaming HuggingFace dataset, returning a list of processed items.
    """
    data_list = []
    s3_client = get_s3_client()
    ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)
    current_processed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_map: dict[Future, int] = {}
        dataset_iterator = iter(ds)
        try:
            # Prime the thread pool
            for i in range(max_workers * 2):
                if stop_processing_flag.is_set():
                    break
                try:
                    file_item = next(dataset_iterator)
                    future = executor.submit(process_file, file_item, s3_client)
                    futures_map[future] = current_processed_count + i
                except StopIteration:
                    log_print("Reached end of dataset stream while submitting initial tasks.")
                    break
                except Exception as e:
                    log_print(f"Error submitting initial task: {e}")

            while futures_map and not stop_processing_flag.is_set():
                for future in as_completed(list(futures_map)):
                    original_idx = futures_map.pop(future)
                    if stop_processing_flag.is_set():
                        if not future.done():
                            future.cancel()
                        continue
                    try:
                        result = future.result()
                    except Exception as exc:
                        log_print(f"Task for item originally at index ~{original_idx} generated an exception: {exc}")
                        result = None

                    if result:
                        with lock:
                            code_file = {
                                "fileId": current_processed_count,
                                "blobId": result["blobId"],
                                "directoryId": result["directoryId"],
                                "path": result["path"],
                                "contentId": result["contentId"],
                                "detectedLicenses": result["detectedLicenses"],
                                "licenseType": result["licenseType"],
                                "repoName": result["repoName"],
                                "githubId": result["githubId"],
                                "language": result["language"],
                                "content": result["content"],
                                "lengthBytes": result["lengthBytes"],
                                "extension": result["extension"],
                                "piiRecords": []
                            }
                            data_list.append(code_file)
                            current_processed_count += 1
                            if current_processed_count % 1000 == 0:
                                log_print(f"Processed {current_processed_count} files")
                            # Only set the stop flag and break if we've reached the target
                            if current_processed_count >= num_samples:
                                log_print(f"Reached the maximum number of files to save ({num_samples}). Stopping further processing.")
                                stop_processing_flag.set()
                                break

                    # Submit new tasks if not stopping and not at end
                    if not stop_processing_flag.is_set():
                        try:
                            file_item = next(dataset_iterator)
                            new_future = executor.submit(process_file, file_item, s3_client)
                            futures_map[new_future] = current_processed_count + len(futures_map)
                        except StopIteration:
                            log_print("Reached end of dataset stream. No more tasks to submit.")
                            break
                        except Exception as e:
                            log_print(f"Error submitting new task: {e}")

                if stop_processing_flag.is_set():
                    break

        except KeyboardInterrupt:
            log_print("\nKeyboard interrupt received. Signaling threads to stop...")
            stop_processing_flag.set()
        finally:
            if stop_processing_flag.is_set():
                log_print("Attempting to cancel remaining tasks...")
                for fut in futures_map:
                    fut.cancel()
            log_print("Shutting down ThreadPoolExecutor. Waiting for running tasks to complete (or be cancelled)...")
            executor.shutdown(wait=True)
            log_print("ThreadPoolExecutor shut down.")

    # If we exit the loop without hitting num_samples, return what we have
    if data_list:
        log_print(f"only downloaded {len(data_list)} items")
        return data_list
    else:
        log_print(f"no items downloaded")
        return None