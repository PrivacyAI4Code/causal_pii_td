# Project Instructions

This repository provides the implementation, datasets, and supplementary materials for paper: A Causal Perspective on the Role of Training Dynamics for Interpreting Privacy Risks in Code Models

## Additional Tables
Supplementary tables are available in **`./extra_tables/`** for reference and reproducibility.  

## Environment Setup
1. Create a Python environment (tested with **Python 3.12**).  
2. Install the required dependencies (adjusting the CUDA version as appropriate):  
   ```bash
   python -m pip install -r requirements.txt
   ```  
Each model is configured through its corresponding `.yaml` file under **`./configs/`**. For illustration, we highlight the configuration for the **StableCode-3B** model.  

## Dataset
We provide the **PII dataset** used in the paper under **`./dataset_all/`**. To construct a customized dataset using our pipeline, modify the configuration file and run:  
```bash
python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task dataset
```  

## Fine-Tuning
Model fine-tuning can be performed with the following command:  
```bash
python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task finetune
```  

## Evaluation
**RQ1 Learning Difficulty Across PII Types:**  
```bash
python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task post_eval
```

**RQ2 Learning Difficulty–Leakage Risk Relationship:**  
```bash
python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task post_attack
```
**RQ3 Causal Effect of Learning Dynamics on Leakage Risk:**

We provide the scm process at ./causal/causal_notebooks

