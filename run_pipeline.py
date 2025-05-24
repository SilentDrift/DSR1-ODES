#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command with progress output."""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"{'='*80}\n")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete ODE solver pipeline")
    parser.add_argument("--train_size", type=int, default=300, help="Size of training dataset")
    parser.add_argument("--eval_size", type=int, default=100, help="Size of evaluation dataset")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-r1-7b-base", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--skip_dataset", action="store_true", help="Skip dataset generation if files exist")
    parser.add_argument("--skip_solving", action="store_true", help="Skip solving phase")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--run_finetune", action="store_true", help="Run LoRA fine-tuning")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_dataset = os.path.join(args.output_dir, "ode_train.json")
    eval_dataset = os.path.join(args.output_dir, "ode_eval.json")
    solutions_file = os.path.join(args.output_dir, "solutions.json")
    eval_output_dir = os.path.join(args.output_dir, "evaluation")
    
    if not args.skip_dataset and (not os.path.exists(train_dataset) or not os.path.exists(eval_dataset)):
        cmd = [
            sys.executable,
            "dataset_builder.py",
            "--train_size", str(args.train_size),
            "--eval_size", str(args.eval_size),
            "--train_output", train_dataset,
            "--eval_output", eval_dataset,
            "--train_seed", str(args.seed),
            "--eval_seed", str(args.seed + 1)
        ]
        if not run_command(cmd, "Generating ODE datasets"):
            print("Dataset generation failed. Exiting.")
            return 1
    
    if not args.skip_solving:
        cmd = [
            sys.executable,
            "solver.py",
            "--file", eval_dataset,
            "--output", solutions_file,
            "--model", args.model,
            "--temperature", str(args.temperature),
            "--half"  # Use half precision for faster inference
        ]
        if not run_command(cmd, "Solving ODEs with DeepSeek-R1"):
            print("ODE solving failed. Exiting.")
            return 1
    
    if not args.skip_evaluation:
        cmd = [
            sys.executable,
            "evaluate.py",
            "--results", solutions_file,
            "--dataset", eval_dataset,
            "--output_dir", eval_output_dir
        ]
        if not run_command(cmd, "Evaluating ODE solutions"):
            print("Evaluation failed. Exiting.")
            return 1
    
    if args.run_finetune:
        finetune_output_dir = os.path.join(args.output_dir, "finetune")
        cmd = [
            sys.executable,
            "finetune_lora.py",
            "--train_dataset", train_dataset,
            "--eval_dataset", eval_dataset,
            "--model_name", args.model,
            "--output_dir", finetune_output_dir,
            "--seed", str(args.seed)
        ]
        if not run_command(cmd, "Fine-tuning with LoRA"):
            print("Fine-tuning failed. Exiting.")
            return 1
        
        finetuned_solutions = os.path.join(args.output_dir, "finetuned_solutions.json")
        finetuned_model = os.path.join(finetune_output_dir)
        
        cmd = [
            sys.executable,
            "solver.py",
            "--file", eval_dataset,
            "--output", finetuned_solutions,
            "--model", finetuned_model,
            "--temperature", str(args.temperature),
            "--half"
        ]
        if not run_command(cmd, "Solving ODEs with fine-tuned model"):
            print("Fine-tuned model evaluation failed. Exiting.")
            return 1
        
        finetuned_eval_dir = os.path.join(args.output_dir, "finetuned_evaluation")
        cmd = [
            sys.executable,
            "evaluate.py",
            "--results", finetuned_solutions,
            "--dataset", eval_dataset,
            "--output_dir", finetuned_eval_dir
        ]
        if not run_command(cmd, "Evaluating fine-tuned model"):
            print("Fine-tuned model evaluation failed. Exiting.")
            return 1
    
    print("\n\nPipeline completed successfully!")
    print(f"Results are available in: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 