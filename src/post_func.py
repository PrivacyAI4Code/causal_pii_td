# from sympy import N  # Unused import, commented out
import os
# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import random
from itertools import chain
import numpy as np
from tqdm import tqdm

from utils import log_print


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""
    model_name_or_path: Optional[str] = field(default=None) # The model checkpoint for weights initialization. None if training from scratch.
    model_type: Optional[str] = field(default=None) # The model type to use when training from scratch.
    config_overrides: Optional[str] = field(default=None) # Override some existing config settings when a model is trained from scratch.
    config_name: Optional[str] = field(default=None) # The name of the model config to use. None if using the same as model_name_or_path.
    tokenizer_name: Optional[str] = field(default=None) # The name of the tokenizer to use. None if using the same as model_name_or_path.
    cache_dir: Optional[str] = field(default=None) # Where to store the pretrained models downloaded from huggingface.
    use_fast_tokenizer: bool = field(default=True) # Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.
    model_revision: str = field(default="main") # The specific model version to use (can be a branch name, tag name or commit id).
    token: str = field(default="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") # The token to use to access the model.
    trust_remote_code: bool = field(default=False) # Whether to trust remote code when downloading a model.
    torch_dtype: Optional[str] = field(default=None) # ["auto", "bfloat16", "float16", "float32"]
    pad_to_multiple_of: bool = field(default=False) # "Whether to pad the embedding layer to a multiple depending on the device. ","For NVIDIA GPUs, this will be a multiple of 8, for TPUs a multiple of 128.",
    attn_implementation: Optional[str] = field(default="sdpa") # "The attention implementation to use. "
    use_lora: bool = field(default=False) # Whether to use LoRA.
    lora_r: int = field(default=16) # LoRA rank. Higher values allow more expressiveness but use more parameters.
    lora_alpha: int = field(default=32) # LoRA alpha parameter for scaling.
    lora_dropout: float = field(default=0.1) # LoRA dropout rate.
    lora_target_modules: Optional[str] = field(default=None) # Comma-separated list of target modules for LoRA. If None, will use default modules for the model architecture.
    save_every_epoch: bool = field(default=False) # Whether to save model checkpoint after every epoch.
    hub_name: Optional[str] = field(default=None) # The name of the hub to push the model to.
    experiment_name: Optional[str] = field(default=None) # The name of the experiment.
    push_to_hub_every_epoch: bool = field(default=False) # Whether to push the model to the hub every epoch.

    num_train_epochs: Optional[int] = field(default=10)

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None) # The name of the dataset to use (via the datasets library).
    dataset_config_name: Optional[str] = field(default=None) # "The configuration name of the dataset to use (via the datasets library)."
    train_file: Optional[str] = field(default=None) # The input training data file (a text file) or a json file with the following format: {"text": "..."}
    validation_file: Optional[str] = field(default=None) # An optional input evaluation data file to evaluate the perplexity on (a text file) or a json file with the following format: {"text": "..."}
    # test_file: Optional[str] = field(default=None) # An optional input test data file to evaluate the perplexity on (a text file) or a json file with the following format: {"text": "..."}
    #dynamic_file: Optional[str] = field(default=None) # An optional input dynamic data file to evaluate the perplexity on (a text file) or a json file with the following format: {"text": "..."}
    max_train_samples: Optional[int] = field(default=None) # For debugging purposes or quicker training, truncate the number of training examples to this value if set.
    max_eval_samples: Optional[int] = field(default=None) # For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.
    streaming: bool = field(default=False) # Enable streaming mode
    block_size: Optional[int] = field(default=None) # Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).
    fim_rate: Optional[float] = field(default=0.5) # Optional probability with which the FIM transformation is applied to the example. Default is 0.5. A rate of 1.0 means every example will undergo FIM transformation, while a rate of 0.0 means no example will.
    fim_spm_rate: Optional[float] = field(default=0.5) # Within the examples undergoing FIM transformation, this rate determines the probability of applying the Sentence Permutation Mode (SPM). Default is 0.5. A rate of 1.0 means all FIM transformations will use SPM, while a rate of 0.0 means none will.
    truncate_or_pad: Optional[bool] = field(default=True) # Indicates whether the transformed example should be truncated or padded to maintain the same length as the original example. Default is True. If False, the function will not truncate or pad the examples.
    fim_prefix_token: Optional[str] = field(default="<fim_prefix>") # Fill-in-Middle Prefix token. Defaults to '<fim_prefix>'.
    fim_middle_token: Optional[str] = field(default="<fim_middle>") # Fill-in-Middle Middle token. Defaults to '<fim_middle>'.
    fim_suffix_token: Optional[str] = field(default="<fim_suffix>") # Fill-in-Middle Suffix token. Defaults to '<fim_suffix>'.
    pad_token: Optional[str] = field(default="<fim_pad>") # Fill-in-Middle Pad token. Used only when 'truncate_or_pad' is set to True. Defaults to '<fim_pad>'.
    overwrite_cache: bool = field(default=False) # Overwrite the cached training and evaluation sets
    validation_split_percentage: Optional[int] = field(default=5) # The percentage of the train set used as validation set in case there's no validation split
    preprocessing_num_workers: Optional[int] = field(default=None) # The number of processes to use for the preprocessing.
    keep_linebreaks: bool = field(default=True) # Whether to keep line breaks when using TXT files or not.
    using_fim: bool = field(default=True) # Whether to use FIM.

    test_file: Optional[str] = field(default=None)
    dynamic_file: Optional[str] = field(default=None)
    post_eval_batch_size: int = field(default=1)
    post_attack_batch_size: int = field(default=8)
    prompt_missing_size: int = field(default=50)
    seed: Optional[int] = field(default=42)

def _ensure_fim_special_tokens(model, tokenizer, fim_prefix_token, fim_middle_token, fim_suffix_token, device_is_cuda, truncate_or_pad=False, pad_token=None):    
    special_tokens = [
        fim_prefix_token,
        fim_middle_token,
        fim_suffix_token,
    ]
    if truncate_or_pad:
        special_tokens.append(pad_token)

    added = tokenizer.add_tokens(special_tokens)
    if added and added > 0:
        pad_factor = 8 if device_is_cuda else 1
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_factor)
    return tokenizer

def _to_torch_dtype(s):
    if s in ["auto", None]:
        return "auto"
    return getattr(torch, s)

def batch_iterator(items, batch_size):
    """Simple batch iterator yielding lists of size up to batch_size."""
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def build_checkpoint_paths(model_args, data_args):
    assert model_args.hub_name is not None
    assert model_args.experiment_name is not None
    assert model_args.num_train_epochs is not None
    assert data_args.using_fim, "not using fim is not supported"
    hub_name = model_args.hub_name
    base_model = model_args.model_name_or_path.split("/")[-1]
    task = "fim" if data_args.using_fim else "ft"
    ft_type = "lora" if model_args.use_lora else "full"
    exp_name = f"{hub_name}/{model_args.experiment_name}" if hub_name else model_args.experiment_name
    paths = []
    paths.append(['base_model', model_args.model_name_or_path])
    for epoch in range(1, model_args.num_train_epochs + 1):
        version = f"{base_model}-{task}-{ft_type}-epoch-{int(epoch)}"
        repo_id = exp_name
        checkpoint_path = f"{repo_id}@{version}"
        paths.append([f"epoch-{int(epoch)}", checkpoint_path])
        log_print(f"will check: {checkpoint_path} from hub")
    return paths

def load_model_and_tokenizer(checkpoint_path, model_args, data_args, using_lora=True, using_fim=False, base_model=None, base_tokenizer=None, mode="test"):
    version = "main"
    # if mode == "attack":
    #     padding_side = "left"
    # else:
    #     padding_side = "right"
        

    if "@" in checkpoint_path: # means the checkpoint_path has a revision
        repo_id, version = checkpoint_path.split("@")
        log_print(f"Loading model and tokenizer from {repo_id} with version {version}")
    else:
        repo_id = checkpoint_path
        log_print(f"Loading model and tokenizer from {checkpoint_path}")
    
    if using_lora:
        log_print("Loading base model and tokenizer for LoRA evaluation...")
        
        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=_to_torch_dtype(model_args.torch_dtype),
                attn_implementation=model_args.attn_implementation,
                )
        else:
            log_print(f"Base model already loaded, using {base_model.config.name_or_path}")
        
        base_tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                revision=version,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                
            )

        log_print(f"Loaded tokenizer from checkpoint: {repo_id} with version {version}")

        model = PeftModel.from_pretrained(base_model, repo_id, revision=version)
        tokenizer = base_tokenizer
    
    else:
        log_print("Loading model and tokenizer from checkpoint for non-LoRA evaluation...")
        config = AutoConfig.from_pretrained(
            repo_id,
            revision=version,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=version,
            config=config,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=_to_torch_dtype(model_args.torch_dtype),
            attn_implementation=model_args.attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=version,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            
        )
    device_is_cuda = torch.cuda.is_available()
    if using_fim:
        tokenizer = _ensure_fim_special_tokens(model, tokenizer, data_args.fim_prefix_token, data_args.fim_middle_token, data_args.fim_suffix_token, device_is_cuda, data_args.truncate_or_pad, data_args.pad_token) 
    
    # Set up tokenizer defaults to avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = data_args.pad_token if hasattr(data_args, 'pad_token') and data_args.pad_token else tokenizer.eos_token
    
    # Set padding side based on mode to avoid warnings during data processing
    
    
    return model, tokenizer

def _fim_transform(example, data_args, np_rng, pad_tok_id, prefix_tok_id, middle_tok_id, suffix_tok_id):
    if np_rng and np_rng.binomial(1, data_args.fim_rate):
        boundaries = sorted(np_rng.randint(low=0, high=len(example) + 1, size=2))

        prefix = example[: boundaries[0]]
        middle = example[boundaries[0] : boundaries[1]]
        suffix = example[boundaries[1] :]

        if data_args.truncate_or_pad:
            total_length = len(prefix) + len(middle) + len(suffix) + 3
            diff = total_length - len(example)
            if diff > 0:
                suffix = suffix[: max(0, len(suffix) - diff)]
            elif diff < 0:
                suffix.extend([pad_tok_id] * (-diff))

        if np_rng.binomial(1, data_args.fim_spm_rate):
            # Apply Suffix-Prefix-Middle (SPM) transformation
            transformed_example = [suffix_tok_id] + suffix + [prefix_tok_id] + prefix + [middle_tok_id] + middle
        else:
            # Apply Prefix-Suffix-Middle (PSM) transformation
            transformed_example = [prefix_tok_id] + prefix + [suffix_tok_id] + suffix + [middle_tok_id] + middle
    else:
        transformed_example = example

    return transformed_example

def load_dataset_for_post(dataset_path, data_args, block_size, tokenizer, mode = "test", verbose=True, np_rng=None):
    random.seed(data_args.seed)
    assert dataset_path.endswith('.json'), "Dataset must be a JSON file"
    assert mode in ["test", "dynamic", "attack"], "Mode must be either test or dynamic or attack"
    
    # Set up tokenizer padding early to avoid warnings during data processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = data_args.pad_token if hasattr(data_args, 'pad_token') and data_args.pad_token else tokenizer.eos_token
    
    
    # Initialize FIM token IDs - always define them to avoid scope issues
    prefix_tok_id = None
    middle_tok_id = None
    suffix_tok_id = None
    pad_tok_id = None
    if data_args.truncate_or_pad:
        pad_tok_id = tokenizer.convert_tokens_to_ids(data_args.pad_token)
    if data_args.using_fim:
        # Get the FIM-specific token ids
        prefix_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_prefix_token)
        middle_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_middle_token)
        suffix_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_suffix_token)

    
    # load the dataset from json file
    dataset_tmp = []
    dataset_tmp = json.load(open(dataset_path, "r"))
    
    final_dataset = []
    # middle_tokens_map = {}
    for idx, data in enumerate(dataset_tmp):
        location_start = data["location_start"]
        location_end = data["location_end"]
        example_id = data["id"]
        pii_type = data["piiType"]
        
        prefix_text = data["text"][:location_start]
        middle_text = data["text"][location_start:location_end]
        suffix_text = data["text"][location_end:]

        if verbose and idx < 3:
            #log_print(f"prefix_text: {prefix_text[-50:]}")
            log_print(f"middle_text: {middle_text}")
            #log_print(f"suffix_text: {suffix_text[:50]}")
        
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        middle_tokens = tokenizer.encode(middle_text, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)

        if mode == "test":
            input_ids = prefix_tokens + middle_tokens + suffix_tokens
            if data_args.using_fim:
                input_ids = _fim_transform(input_ids, data_args, np_rng, pad_tok_id, prefix_tok_id, middle_tok_id, suffix_tok_id)
            #target_positions = [1] * len(input_ids) # for test, we care every token
            final_dataset.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "example_id": example_id,
                "pii_type": pii_type,
                #"target_positions": target_positions,
            })
        elif mode == "dynamic":
            budget = block_size - len(middle_tokens)
            
            if data_args.using_fim:
                budget -= 3
            if budget <= 0:
                raise ValueError(f"budget is negative: {budget}")
            if data_args.using_fim:
                prefix_budget = budget // 2
                if len(prefix_tokens) > prefix_budget:
                    prefix_tokens = prefix_tokens[-prefix_budget:]
                suffix_budget = budget - len(prefix_tokens)
                if len(suffix_tokens) > suffix_budget:
                    suffix_tokens = suffix_tokens[:suffix_budget]

                if np_rng.binomial(1, data_args.fim_spm_rate):
                    # SPM: suffix_tok_id + suffix + prefix_tok_id + prefix + middle_tok_id + middle
                    input_ids = [suffix_tok_id] + suffix_tokens + [prefix_tok_id] + prefix_tokens + [middle_tok_id] + middle_tokens
                    target_position = 1 + len(suffix_tokens) + 1 + len(prefix_tokens) + 1  # position after middle_tok_id
                else:
                    # PSM: prefix_tok_id + prefix + suffix_tok_id + suffix + middle_tok_id + middle
                    input_ids = [prefix_tok_id] + prefix_tokens + [suffix_tok_id] + suffix_tokens + [middle_tok_id] + middle_tokens
                    target_position = 1 + len(prefix_tokens) + 1 + len(suffix_tokens) + 1  # position after middle_tok_id
            else:
                raise ValueError(f"not implemented when using_fim is False and mode is dynamic")
            
            assert len(input_ids) <= block_size
            # target_positions = [0] * len(target_position) + [1] * len(middle_tokens)
            final_dataset.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "example_id": example_id,
                "pii_type": pii_type,
                "middle_tokens": middle_tokens,
                "middle_start": target_position-1,
            })
        elif mode == "attack":
            prompt_missing_size = data_args.prompt_missing_size
            # if pii_type not in middle_tokens_map.keys():
            #     middle_tokens_map[pii_type] = set()
            # middle_tokens_map[pii_type].add(middle_tokens)
            
            # split the prompt_missing_size into prefix_missing_size and suffix_missing_size randomly
            prefix_missing_size = random.randint(0, prompt_missing_size)
            suffix_missing_size = prompt_missing_size - prefix_missing_size
            
            budget = block_size - len(middle_tokens)
            if data_args.using_fim:
                budget -= 3
            if budget <= 0:
                raise ValueError(f"budget is negative: {budget}")
            
            if data_args.using_fim:
                prefix_budget = budget // 2

                if len(prefix_tokens) > prefix_budget:
                    prefix_tokens = prefix_tokens[-prefix_budget:]
                suffix_budget = budget - len(prefix_tokens)
                if len(suffix_tokens) > suffix_budget:
                    suffix_tokens = suffix_tokens[:suffix_budget]

                if prefix_missing_size > 0:
                    prefix_tokens = prefix_tokens[:-prefix_missing_size] # remove the prefix tokens at the end
                if suffix_missing_size > 0:
                    suffix_tokens = suffix_tokens[suffix_missing_size:] # remove the suffix tokens at the beginning

                if np_rng.binomial(1, data_args.fim_spm_rate):
                    # SPM: suffix_tok_id + suffix + prefix_tok_id + prefix + middle_tok_id
                    input_ids = [suffix_tok_id] + suffix_tokens + [prefix_tok_id] + prefix_tokens + [middle_tok_id] 
                    target_position = 1 + len(suffix_tokens) + 1 + len(prefix_tokens) + 1  # position after middle_tok_id
                else:
                    # PSM: prefix_tok_id + prefix + suffix_tok_id + suffix + middle_tok_id
                    input_ids = [prefix_tok_id] + prefix_tokens + [suffix_tok_id] + suffix_tokens + [middle_tok_id] 
                    target_position = 1 + len(prefix_tokens) + 1 + len(suffix_tokens) + 1  # position after middle_tok_id
            else:
                raise ValueError(f"not implemented when using_fim is False and mode is attack")
            
            #target_positions = [0] * len(target_position) + [1] * len(middle_tokens)
            final_dataset.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "example_id": example_id,
                "pii_type": pii_type,
                "middle_tokens": middle_tokens,
                "middle_start": target_position,
            })
        
    return final_dataset

def group_texts(examples, block_size):
    """
    Concatenate all texts from our dataset and generate chunks of block_size.
    This is the standard HuggingFace approach for grouping texts for language modeling.
    
    Args:
        examples: List of dictionaries, each containing 'input_ids', 'attention_mask', etc.
        block_size: Maximum sequence length for each chunk
        
    Returns:
        Dictionary with same keys as input, but values are lists of chunks
    """
    # Handle empty input
    if not examples:
        return {}
    
    # Get all the keys from the first example (assuming all examples have same keys)
    keys = examples[0].keys()
    
    # Concatenate all the texts for each key
    concatenated_examples = {}
    for key in keys:
        if key in ['input_ids', 'attention_mask']:  # Only process sequence data
            concatenated_examples[key] = []
            for example in examples:
                concatenated_examples[key].extend(example[key])
    
    # Handle case where we don't have enough data
    total_length = len(concatenated_examples['input_ids'])
    if total_length < block_size:
        return {}
    
    # We drop the small remainder, we could add padding if the model supported it instead of this drop
    total_length = (total_length // block_size) * block_size
    
    # Split by chunks of max_len
    result = {}
    for key in concatenated_examples.keys():
        result[key] = [
            concatenated_examples[key][i : i + block_size] 
            for i in range(0, total_length, block_size)
        ]
    
    # For language modeling, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    
    return result

def collate_fim_examples(
    examples: List[Dict[str, Any]],
    pad_token_id: int,
    label_pad_id: int = -100,
) -> Dict[str, torch.Tensor]:
    """Pad a list of variable-length FIM examples into a batch.

    Each example is a dict with keys:
      - input_ids: List[int]
      - labels: List[int]
    Optional:
      - attention_mask: List[int]
    """
    max_len = max(len(e["input_ids"]) for e in examples)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    for e in examples:
        inp = e["input_ids"]
        lab = e["labels"] if "labels" in e else e["input_ids"]
        pad_inp = inp + [pad_token_id] * (max_len - len(inp))
        # labels 对齐，但 pad 区域设为 -100，以便在度量中忽略
        pad_lab = lab + [label_pad_id] * (max_len - len(lab))
        attn = [1] * len(inp) + [0] * (max_len - len(inp))
        batch_input_ids.append(pad_inp)
        batch_labels.append(pad_lab)
        batch_attention_mask.append(attn)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
    }

@torch.no_grad()
def fim_generate_batched(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    block_size: int,
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> Tuple[List[str], List[List[int]]]:
    """基于 FIM 提示，逐条生成 middle 文本与 tokens。

    返回：
        texts:  每条样本生成的 middle 文本
        tokens: 对应的 token 序列
    """
    device = next(model.parameters()).device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if attention_mask is None:
        attention_mask = (input_ids != pad_token_id).long()
    else:
        attention_mask = attention_mask.to(device)

    B = input_ids.size(0)
    texts: List[str] = []
    tokens: List[List[int]] = []

    fim_end_id = tokenizer.convert_tokens_to_ids("<fim_end>")
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    if fim_end_id is not None and fim_end_id != tokenizer.eos_token_id:
        eos_ids.append(fim_end_id)
    eos_ids = eos_ids or None

    # 逐条生成，避免批量限制
    for i in range(B):
        # 获取单条样本
        single_input_ids = input_ids[i:i+1]
        single_attention_mask = attention_mask[i:i+1]
        
        # 计算该样本的剩余上下文长度
        prompt_len = int(single_attention_mask.sum().item())
        model_max = getattr(model.config, "max_position_embeddings", None)
        hard_limit = model_max or block_size
        remaining_ctx = max(0, hard_limit - prompt_len)
        block_remaining = max(0, block_size - prompt_len)
        max_new_tokens = min(remaining_ctx, block_remaining, 512)  # 限制最大生成长度
        
        if max_new_tokens <= 0:
            # 如果没有剩余空间，返回空结果
            texts.append("")
            tokens.append([])
            continue

        # 单条生成
        outputs = model.generate(
            input_ids=single_input_ids,
            attention_mask=single_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_ids,
            use_cache=True,
            return_dict_in_generate=True,
        )

        # 处理生成结果
        seq = outputs.sequences[0]
        new_tok = seq[prompt_len:]
        text = tokenizer.decode(new_tok, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        # 停止条件处理
        for stopper in ["<fim_end>", tokenizer.eos_token, "<|endoftext|>"]:
            if stopper and stopper in text:
                text = text.split(stopper, 1)[0]
                break
        
        toks = tokenizer.encode(text, add_special_tokens=False)
        texts.append(text)
        tokens.append(toks)

    return texts, tokens

@torch.no_grad()
def evaluate_model_on_dataset(model, tokenizer, dataset, data_args, device, mode, np_rng=None):
    assert mode in ["test", "dynamic", "attack"], "Mode must be either test or dynamic or attack"
    log_print(f"Evaluating on {len(dataset)} examples under {mode} mode...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(device)
    model.eval()

    # Set up tokenizer padding for generation tasks
    if tokenizer.pad_token is None:
        tokenizer.pad_token = data_args.pad_token if hasattr(data_args, 'pad_token') and data_args.pad_token else tokenizer.eos_token
    
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    total_tokens = 0
    correct_tokens = 0
    summary_info = []

    if mode == "test":
        grouped_dataset = group_texts(dataset, data_args.block_size)
        if not grouped_dataset:  # Handle empty dataset
            log_print("Warning: grouped_dataset is empty, skipping evaluation")
            return [], None
            
        # Convert grouped format to list of examples
        num_examples = len(grouped_dataset["input_ids"])
        examples_list = []
        for i in range(num_examples):
            example = {k: v[i] for k, v in grouped_dataset.items()}
            examples_list.append(example)
        
        # do batch evaluation
        with torch.no_grad():
            for batch in batch_iterator(examples_list, data_args.post_eval_batch_size):
                bt = collate_fim_examples(batch, pad_token_id=pad_token_id, label_pad_id=-100)
                input_ids = bt["input_ids"].to(device)
                labels = bt["labels"].to(device)
                attention_mask = bt["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_tokens = torch.argmax(logits, dim=-1)
                shift_predictions = predicted_tokens[..., :-1].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                flat_predictions = shift_predictions.view(-1)
                flat_labels = shift_labels.view(-1)
                valid_tokens = flat_labels != -100
                if valid_tokens.any():
                    correct = (flat_predictions[valid_tokens] == flat_labels[valid_tokens]).sum().item()
                    total = int(valid_tokens.sum().item())
                    correct_tokens += correct
                    total_tokens += total

        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        log_print(f"Test evaluation - Overall token accuracy: {accuracy:.4f} ({correct_tokens}/{total_tokens})")
        summary_info={
            "accuracy": accuracy,
            "total_tokens": total_tokens,
            "correct_tokens": correct_tokens,
        }
        return summary_info, None
    
    elif mode == "dynamic":
        detailed_predictions = []
        total_tokens_by_type = {}
        correct_tokens_by_type = {}

        # do batch evaluation
        # only care about the probability and prediction of the middle tokens
        with torch.no_grad():
            for batch in batch_iterator(dataset, data_args.post_eval_batch_size):
                ex_ids = [ex["example_id"] for ex in batch]
                ex_types = [ex["pii_type"] for ex in batch]
                starts = [ex["middle_start"] for ex in batch]
                
                # Calculate ends based on middle_tokens length
                ends = []
                for ex in batch:
                    start = ex["middle_start"]
                    middle_len = len(ex["middle_tokens"])
                    ends.append(start + middle_len)

                bt = collate_fim_examples(batch, pad_token_id=pad_token_id, label_pad_id=-100)
                input_ids = bt["input_ids"].to(device)
                labels = bt["labels"].to(device)
                attention_mask = bt["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                predicted_tokens = torch.argmax(logits, dim=-1)
                shift_predictions = predicted_tokens[..., :-1].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                B, Lm1 = shift_predictions.shape
                for i in range(B):
                    example_id = ex_ids[i]
                    pii_type_label = ex_types[i]
                    if pii_type_label not in total_tokens_by_type:
                        total_tokens_by_type[pii_type_label] = 0
                    if pii_type_label not in correct_tokens_by_type:
                        correct_tokens_by_type[pii_type_label] = 0

                    s = starts[i]
                    e = ends[i]
                    seq_len = int(attention_mask[i].sum().item())
                    upper = max(0, min(e, seq_len - 1))
                    lower = max(0, min(s, seq_len - 1))
                    if upper <= lower:
                        continue

                    pred_i = shift_predictions[i, lower:upper]
                    lab_i = shift_labels[i, lower:upper]

                    correct = (pred_i == lab_i).sum().item()
                    total = int((lab_i != -100).sum().item())
                    total_tokens_by_type[pii_type_label] += total
                    correct_tokens_by_type[pii_type_label] += correct
                    
                    # Also update global counters
                    total_tokens += total
                    correct_tokens += correct

                    mid_logits = shift_logits[i, lower:upper, :].float().cpu()
                    #pred_probs = torch.softmax(mid_logits, dim=-1).gather(1, pred_i.cpu().unsqueeze(1)).squeeze(1).tolist()
                    true_probs = torch.softmax(mid_logits, dim=-1).gather(1, lab_i.cpu().unsqueeze(1)).squeeze(1).tolist()

                    pred_words = tokenizer.decode(pred_i.cpu().numpy(), skip_special_tokens=False)
                    true_words = tokenizer.decode(lab_i.cpu().numpy(), skip_special_tokens=False)

                    detailed_predictions.append({
                        "example_id": example_id,
                        "type": pii_type_label,
                        "expected_words": true_words,
                        "predicted_words": pred_words,
                        "pred_token_probs": true_probs,
                    })

        # Create summary by type
        dynamic_results = {}
        for pii_type_label in total_tokens_by_type.keys():
            dynamic_results[pii_type_label] = {
                "accuracy": correct_tokens_by_type[pii_type_label] / total_tokens_by_type[pii_type_label] if total_tokens_by_type[pii_type_label] > 0 else 0.0,
                "total_tokens": total_tokens_by_type[pii_type_label],
                "correct_tokens": correct_tokens_by_type[pii_type_label],
            }

        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        log_print(f"Dynamic evaluation - Overall token accuracy: {accuracy:.4f} ({correct_tokens}/{total_tokens})")
        
        summary_info={
            "accuracy": accuracy,
            "total_tokens": total_tokens,
            "correct_tokens": correct_tokens,
            "by_type": dynamic_results,
        }
        return summary_info, detailed_predictions
    
    elif mode == "attack":
        detailed_predictions = []
        generated_count = 0
        
        # do individual generation 
        # only care about the generated tokens
        with torch.no_grad():
            for example in tqdm(dataset, desc="Generating attack examples"):
                example_id = example["example_id"]
                pii_type_label = example["pii_type"]
                expected_tokens = example["middle_tokens"]
                middle_start = example["middle_start"]
                
                # 准备单条样本的输入
                input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)

                # 单条生成
                texts, generated_tokens_list = fim_generate_batched(
                    input_ids=input_ids,
                    model=model,
                    tokenizer=tokenizer,
                    block_size=data_args.block_size,
                    pad_token_id=pad_token_id,
                    attention_mask=attention_mask,
                    temperature=0.1,
                    top_p=0.95,
                )

                # 处理生成结果（应该只有一个结果）
                generated_text = texts[0] if texts else ""
                generated_tokens = generated_tokens_list[0] if generated_tokens_list else []

                expected_text = tokenizer.decode(expected_tokens, skip_special_tokens=False)
                
                detailed_predictions.append({
                    "example_id": example_id,
                    "type": pii_type_label,
                    "expected_text": expected_text,
                    "middle_start": middle_start,
                    #"expected_tokens": expected_tokens,
                    "generated_text": generated_text,
                    #"generated_tokens": generated_tokens,
                })
                generated_count += 1

        log_print(f"Attack generation - Generated {generated_count} examples")
        
        # For attack mode, we don't compute accuracy in the traditional sense
        # Instead we return generation statistics
        summary_info={
            "generated_count": generated_count,
            "total_examples": len(dataset),
            "generation_rate": generated_count / len(dataset) if len(dataset) > 0 else 0.0,
        }
        return summary_info, detailed_predictions