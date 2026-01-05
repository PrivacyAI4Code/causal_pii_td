import json
import os
from collections import Counter

dataset_path = "/home/hyang45/gitV1/pii_final/dataset_all/train_dataset.json"

dataset = json.load(open(dataset_path, "r"))

# if the id is the same, the value is the same?
unique = []
for example in dataset:
    unique.append((example["id"], example["location_start"], example["location_end"]))
# count of each value in unique
count = Counter(unique)
answer = "/home/hyang45/gitV1/answer.txt"
for key, value in count.items():
    if value > 1:
        with open(answer, "a") as f:
            f.write(f"{key}: {value}\n")
    
                
