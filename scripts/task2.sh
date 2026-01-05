HF_TOKEN_VALUE="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"
CONFIG_FILE="bigcode_starcoder2-3b_fim.yaml"

# export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN_VALUE"
# export HF_TOKEN="$HF_TOKEN_VALUE"
# export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VALUE"  # 备用变量名
# export HF_ACCESS_TOKEN="$HF_TOKEN_VALUE"         # 某些工具使用的变量名

# # Configuration
# CONFIGS_DIR="./failed_configs"
# RESULTS_DIR="./results"
# LOGS_DIR="./logs"
# SCRIPT_PATH="../scripts/run_experiment_cgs.py"
# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
#CACHE_DIR="/data/tomasfang/yoloe-test/dit/verl/.cache"

# Set environment variables for caching
# export HF_HOME="$CACHE_DIR"
# export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/hub"
# export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
# export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
# export VLLM_CACHE_DIR="$CACHE_DIR/vllm"
# export XDG_CACHE_HOME="$CACHE_DIR"

CUDA_VISIBLE_DEVICES=0,1 python run.py --config configs/$CONFIG_FILE --task finetune
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/$CONFIG_FILE --task post_eval
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/$CONFIG_FILE --task post_attack