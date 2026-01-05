CUDA_VISIBLE_DEVICES=1 python run.py --config configs/Qwen_Qwen2.5-Coder-14B_fim.yaml --task finetune
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/codellama_CodeLlama-13b-hf_fim.yaml --task finetune
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/bigcode_starcoder2-15b_fim.yaml --task finetune
