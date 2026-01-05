CUDA_VISIBLE_DEVICES=1 python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task finetune
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/Qwen_Qwen2.5-Coder-3B_fim.yaml --task finetune
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/bigcode_starcoder2-3b_fim.yaml --task finetune