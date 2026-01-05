#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task post_eval
#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/Qwen_Qwen2.5-Coder-3B_fim.yaml --task post_eval
#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/bigcode_starcoder2-3b_fim.yaml --task post_eval

CUDA_VISIBLE_DEVICES=0 python run.py --config configs/bigcode_starcoder2-7b_fim.yaml --task post_eval
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/Qwen_Qwen2.5-Coder-7B_fim.yaml --task post_eval
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/codellama_CodeLlama-7b-hf_fim.yaml --task post_eval

CUDA_VISIBLE_DEVICES=0 python run.py --config configs/Qwen_Qwen2.5-Coder-14B_fim.yaml --task post_eval
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/codellama_CodeLlama-13b-hf_fim.yaml --task post_eval
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/bigcode_starcoder2-15b_fim.yaml --task post_eval



#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task post_attack    
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/Qwen_Qwen2.5-Coder-3B_fim.yaml --task post_attack
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/bigcode_starcoder2-3b_fim.yaml --task post_attack

CUDA_VISIBLE_DEVICES=1 python run.py --config configs/bigcode_starcoder2-7b_fim.yaml --task post_attack
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/Qwen_Qwen2.5-Coder-7B_fim.yaml --task post_attack
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/codellama_CodeLlama-7b-hf_fim.yaml --task post_attack

CUDA_VISIBLE_DEVICES=1 python run.py --config configs/Qwen_Qwen2.5-Coder-14B_fim.yaml --task post_attack
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/codellama_CodeLlama-13b-hf_fim.yaml --task post_attack
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/bigcode_starcoder2-15b_fim.yaml --task post_attack
