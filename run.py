import os
import yaml
import argparse
import logging
import subprocess
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(filename)s - %(funcName)s - %(lineno)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="loads config and runs the pipeline")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--task", type=str, required=True, help="task to run")
    args = parser.parse_args()
    
    # Load the config from yaml
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    experiment_name = config['general']['experiment_name']

    experiment_dir = os.path.abspath(f"./save/{experiment_name}")
    os.makedirs(experiment_dir, exist_ok=True)

    

    log_dir = os.path.join(experiment_dir, config['general']['log_dir'])
    os.makedirs(log_dir, exist_ok=True)

    data_dir = os.path.join(experiment_dir, config['general']['data_dir'])
    os.makedirs(data_dir, exist_ok=True)

    ckpt_dir = os.path.join(experiment_dir, config['general']['ckpt_dir'])
    os.makedirs(ckpt_dir, exist_ok=True)

    src_dir = os.path.abspath("./src")
    os.makedirs(src_dir, exist_ok=True)
    
    post_eval_dir = os.path.join(experiment_dir, config['general']['post_eval_dir'])
    os.makedirs(post_eval_dir, exist_ok=True)

    #log_file = os.path.join(log_dir, f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['model']['model_name_or_path'].replace('/', '_')}_{config['dataset']['source_name'].replace('/', '_')}_{config['dataset']['subset']}.log")
    log_file = os.path.join("log_all", f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['model']['model_name_or_path'].replace('/', '_')}_{config['dataset']['source_name'].replace('/', '_')}_{config['dataset']['subset']}.log")
    
    fh = logging.FileHandler(log_file)
    logger.addHandler(fh)
    log_fh = open(os.path.abspath(log_file), "a")

    # Load the environment variables
    load_dotenv()

    current_dir = os.getcwd()
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"

    # 1. dataset
    if args.task == "dataset":
        run_code = [
            "python",
            "dataset.py",
            "--config", os.path.abspath(args.config),
            "--output_dir", os.path.abspath(data_dir)
        ]
        logger.info(f"Running code: {run_code}")
        subprocess.run(run_code, cwd=src_dir, env=env, stdout=log_fh, stderr=log_fh)

    # 2. finetune
    elif args.task == "finetune":
        run_code = [
            "python",
            "ft_fim.py",
            "--config", os.path.abspath(args.config),
            "--output_dir", os.path.abspath(ckpt_dir),
            #"--train_file", os.path.abspath(os.path.join(data_dir, "train_dataset.json")),
            "--train_file", os.path.abspath("dataset_all/train_dataset.json"),
            "--validation_file", os.path.abspath("dataset_all/val_dataset.json"),
            #"--validation_file", os.path.abspath(os.path.join(data_dir, "val_dataset.json")),
            #"--test_file", os.path.abspath(os.path.join(data_dir, experiment_name, "test_dataset.json"))
        ]
        logger.info(f"Running code: {run_code}")
        subprocess.run(run_code, cwd=src_dir, env=env, stdout=log_fh, stderr=log_fh)

    # 3. post_eval
    elif args.task == "post_eval":
        run_code = [
            "python",
            "post_eval.py",
            "--config", os.path.abspath(args.config),
            "--output_dir", os.path.abspath(post_eval_dir),
            "--test_file", os.path.abspath("dataset_all/test_dataset.json"),
            "--dynamic_file", os.path.abspath("dataset_all/train_dataset.json")
            
        ]
        logger.info(f"Running code: {run_code}")
        subprocess.run(run_code, cwd=src_dir, env=env, stdout=log_fh, stderr=log_fh)

    # 4. pii attack
    elif args.task == "post_attack":
        run_code = [
            "python",
            "post_attack.py",
            "--config", os.path.abspath(args.config),
            "--output_dir", os.path.abspath(post_eval_dir),
            "--attack_file", os.path.abspath("dataset_all/train_dataset.json"),
            
        ]
        logger.info(f"Running code: {run_code}")
        subprocess.run(run_code, cwd=src_dir, env=env, stdout=log_fh, stderr=log_fh)
    
    # 5. post_eval_train
    elif args.task == "post_eval_train":
        run_code = [
            "python",
            "post_eval_train.py",
            "--config", os.path.abspath(args.config),
            "--output_dir", os.path.abspath(post_eval_dir),
            "--test_file", os.path.abspath("dataset_all/train_dataset.json"),
            "--dynamic_file", "None"
        ]
        logger.info(f"Running code: {run_code}")
        subprocess.run(run_code, cwd=src_dir, env=env, stdout=log_fh, stderr=log_fh)

    # 5. casual

if __name__ == "__main__":
    main()
    
    
    