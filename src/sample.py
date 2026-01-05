import random
from collections import defaultdict
import os

from visualize import visualize_dataset
from utils import log_print

def enrich_with_content(records, base_dataset, max_num=None):
    enriched = []
    for i, record in enumerate(records):
        if max_num is not None and i >= max_num:
            break
        rid = record.get("id")
        record_copy = dict(record)  # avoid modifying original
        record_copy["text"] = base_dataset[rid]["content"]
        enriched.append(record_copy)
    return enriched

def sample_dataset(base_dataset, pii_dataset, n_train, n_val, n_test, visualize_size, output_dir):
    dataset_by_piiType = defaultdict(list)
    for pii in pii_dataset:
        piiType = pii.get("piiType")
        dataset_by_piiType[piiType].append(pii)
    
        
    train_dataset = []
    val_dataset = []
    test_dataset = []
    n_train_val = n_train + n_val
    n_test_more = int(n_test * 1.2)

    used_ids = set()
    pii_types_order = ["ip_address", "key", "name", "email", "username", "password"]
    for piiType in pii_types_order:
        selected = []
        for record in dataset_by_piiType.get(piiType, []):
            selected.append(record)
        # if n < n_train_val + n_test_more:
        #     raise ValueError(f"Not enough {piiType} samples to sample. {n} < {n_train_val + n_test_more}")
        
        selected = sorted(selected, key=lambda x: x.get("score", 0), reverse=True)
        new_selected = []
        for record in selected:
            if record.get("id") in used_ids:
                continue
            used_ids.add(record.get("id"))
            new_selected.append(record)
            if len(new_selected) >= n_train_val + n_test_more:
                break

        random.seed(42)
        random.shuffle(selected)
        dataset_train_val = selected[:n_train_val]
        dataset_test = selected[n_train_val:]

        dataset_train = dataset_train_val[:n_train]
        dataset_val = dataset_train_val[n_train:]
       
        # add content to the dataset
        dataset_train_val_enriched = enrich_with_content(dataset_train_val, base_dataset)
        dataset_test_enriched = enrich_with_content(dataset_test, base_dataset)

        dataset_train_enriched = enrich_with_content(dataset_train, base_dataset)
        dataset_val_enriched = enrich_with_content(dataset_val, base_dataset)

        # train_val_dataset.extend(dataset_train_val_enriched)
        train_dataset.extend(dataset_train_enriched)
        val_dataset.extend(dataset_val_enriched)
        test_dataset.extend(dataset_test_enriched)

        dataset_train_val_visualize = dataset_train_val_enriched[:visualize_size]

        # generate the visualize dataset
        visualize_dataset(dataset_train_val_visualize, output_html=os.path.join(output_dir, f"visualize_{piiType}.html"))
        log_print(f"saved visualize dataset to visualize_{piiType}.html")

        # save id
        id_file_path = os.path.join(output_dir, f"id.txt")
        with open(id_file_path, "a") as f:
            f.write(f"type: {piiType} train_val test \n")
            f.write(",".join(str(r["id"]) for r in dataset_train_val_visualize) + "\n")
        log_print(f"saved id to {id_file_path}")
        
    return train_dataset, val_dataset, test_dataset
