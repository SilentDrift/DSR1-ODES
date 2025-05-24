#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import os
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from sympy import symbols, Function, sympify, latex, dsolve
from sympy.abc import x, y, C1, C2

DEFAULT_SYSTEM_PROMPT = """You are DeepSeekMath, an expert mathematical assistant specializing in solving ordinary differential equations (ODEs).
Your task is to solve the given ODE step-by-step, showing your work clearly.
First, identify the type of ODE. Then, apply the appropriate solution method.
If possible, use SymPy to verify your solution by writing Python code.
At the end, provide the general solution in the format "y = [solution expression]".
"""

def format_example_for_training(example: Dict) -> str:
    """Format an ODE example with system prompt, user input, and assistant output."""
    system_prompt = DEFAULT_SYSTEM_PROMPT
    
    conversation = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    user_prompt = f"Solve the differential equation: {example['ode_text']}"
    conversation += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    
    assistant_response = generate_solution_response(example)
    conversation += f"<|im_start|>assistant\n{assistant_response}<|im_end|>\n"
    
    return conversation

def generate_solution_response(example: Dict) -> str:
    """Generate a solution response with step-by-step explanation and SymPy code."""
    ode_type = example["ode_type"]
    ode_text = example["ode_text"]
    solution_text = example["solution_text"]
    
    sympy_code = generate_sympy_code(example)
    
    if ode_type == "separable":
        explanation = generate_separable_explanation(example)
    elif ode_type == "linear_first_order":
        explanation = generate_linear_explanation(example)
    elif ode_type == "bernoulli":
        explanation = generate_bernoulli_explanation(example)
    elif ode_type == "second_order_constant_coeff":
        explanation = generate_second_order_explanation(example)
    elif ode_type == "cauchy_euler":
        explanation = generate_cauchy_euler_explanation(example)
    else:
        explanation = "I'll solve this differential equation step by step."
    
    response = f"{explanation}\n\nI can verify this using SymPy:\n```python\n{sympy_code}\n```\n\nThe general solution is:\n{solution_text}"
    
    return response

def generate_sympy_code(example: Dict) -> str:
    """Generate SymPy code to solve the ODE."""
    ode_sympy = example["ode_sympy"]
    
    code = f"""from sympy import symbols, Function, dsolve, Eq
from sympy.abc import x

# Define the differential equation
y = Function('y')
# Original equation: {example['ode_text']}
eq = {ode_sympy}

# Solve the ODE
solution = dsolve(eq, y(x))
print(solution)
"""
    return code

def generate_separable_explanation(example: Dict) -> str:
    """Generate explanation for separable ODE."""
    return f"""I'll solve this separable differential equation: {example['ode_text']}

This is a separable equation that can be written in the form g(y) dy = f(x) dx.

Step 1: Separate the variables.
{extract_variables_from_text(example['ode_text'])}

Step 2: Integrate both sides.
∫ g(y) dy = ∫ f(x) dx

Step 3: Solve for y."""

def generate_linear_explanation(example: Dict) -> str:
    """Generate explanation for linear first-order ODE."""
    return f"""I'll solve this first-order linear ODE: {example['ode_text']}

This is in the standard form dy/dx + P(x)y = Q(x).

Step 1: Identify the integrating factor μ(x) = e^∫P(x)dx.

Step 2: Multiply both sides of the equation by μ(x).

Step 3: Notice that the left side becomes the derivative of μ(x)y.

Step 4: Integrate both sides.

Step 5: Solve for y by dividing by μ(x)."""

def generate_bernoulli_explanation(example: Dict) -> str:
    """Generate explanation for Bernoulli ODE."""
    return f"""I'll solve this Bernoulli equation: {example['ode_text']}

This is a Bernoulli equation of the form dy/dx + P(x)y = Q(x)y^n.

Step 1: Make the substitution v = y^(1-n) to transform it into a linear equation.

Step 2: Compute dv/dx in terms of dy/dx.

Step 3: Solve the resulting linear equation for v.

Step 4: Substitute back to find y."""

def generate_second_order_explanation(example: Dict) -> str:
    """Generate explanation for second-order ODE with constant coefficients."""
    return f"""I'll solve this second-order linear ODE with constant coefficients: {example['ode_text']}

Step 1: Find the characteristic equation.

Step 2: Solve for the roots of the characteristic equation.

Step 3: Form the complementary solution based on the roots.

Step 4: If the equation is non-homogeneous, find a particular solution.

Step 5: Form the general solution by combining the complementary and particular solutions."""

def generate_cauchy_euler_explanation(example: Dict) -> str:
    """Generate explanation for Cauchy-Euler equation."""
    return f"""I'll solve this Cauchy-Euler equation: {example['ode_text']}

Cauchy-Euler equations have the form x^2y'' + axy' + by = f(x).

Step 1: Try a solution of the form y = x^r.

Step 2: Substitute into the equation to find the characteristic equation.

Step 3: Find the roots of the characteristic equation.

Step 4: Form the general solution based on the roots."""

def extract_variables_from_text(ode_text: str) -> str:
    if "=" in ode_text:
        left, right = ode_text.split("=")
        return f"{left.strip()} = {right.strip()}\nRearranging to separate variables."
    return "Rearranging the equation to separate variables."

def prepare_dataset(
    train_file: str,
    eval_file: str,
    tokenizer,
    max_length: int = 2048
) -> tuple:
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
    
    # Create formatted examples
    train_texts = [format_example_for_training(example) for example in train_data]
    eval_texts = [format_example_for_training(example) for example in eval_data]
    
    # Function to tokenize and format
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
    
    return train_tokenized, eval_tokenized

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-R1 on ODE solving with LoRA")
    parser.add_argument("--train_dataset", type=str, default="ode_train.json", help="Path to training dataset")
    parser.add_argument("--eval_dataset", type=str, default="ode_eval.json", help="Path to evaluation dataset")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-r1-7b-base", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./ode-solver-lora", help="Directory to save the model")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of LoRA adapters")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps to train for (overrides epochs)")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Preparing datasets...")
    train_dataset, eval_dataset = prepare_dataset(
        args.train_dataset,
        args.eval_dataset,
        tokenizer,
        args.max_length
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,  # Use bfloat16 if available
        tf32=True,  # Use tensor cores
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main() 