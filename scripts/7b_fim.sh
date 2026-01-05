CUDA_VISIBLE_DEVICES=1 python run.py --config configs/bigcode_starcoder2-7b_fim.yaml --task finetune
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/Qwen_Qwen2.5-Coder-7B_fim.yaml --task finetune
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/codellama_CodeLlama-7b-hf_fim.yaml --task finetune
