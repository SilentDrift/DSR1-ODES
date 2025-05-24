#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple

import torch
import sympy as sp
from sympy import symbols, Function, dsolve, simplify, Eq
from sympy.abc import x, C1, C2
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

DEFAULT_SYSTEM_PROMPT = """You are DeepSeekMath, an expert mathematical assistant specializing in solving ordinary differential equations (ODEs).
Your task is to solve the given ODE step-by-step, showing your work clearly.
At the end, provide the general solution in the format "y = [solution expression]".
"""

FEW_SHOT_EXAMPLES = [
    {
        "user": "Solve the differential equation: dy/dx + 2y = x",
        "assistant": """I'll solve this first-order linear ODE: dy/dx + 2y = x

This is in the standard form dy/dx + P(x)y = Q(x) where P(x) = 2 and Q(x) = x.

The integrating factor is μ(x) = e^∫P(x)dx = e^∫2dx = e^(2x)

Multiplying both sides by the integrating factor:
e^(2x) dy/dx + 2e^(2x)y = xe^(2x)

The left side is the derivative of e^(2x)y:
d/dx[e^(2x)y] = xe^(2x)

Integrating both sides:
e^(2x)y = ∫xe^(2x)dx

Using integration by parts:
∫xe^(2x)dx = x(e^(2x)/2) - ∫(e^(2x)/2)dx = (x/2)e^(2x) - (1/4)e^(2x) + C = e^(2x)(x/2 - 1/4) + C

Therefore:
e^(2x)y = e^(2x)(x/2 - 1/4) + C

Dividing by e^(2x):
y = x/2 - 1/4 + Ce^(-2x)

The general solution is:
y = x/2 - 1/4 + C1*e^(-2x)"""
    },
    {
        "user": "Solve the ODE: y'' - 3y' + 2y = 0",
        "assistant": """I'll solve this second-order linear homogeneous ODE with constant coefficients: y'' - 3y' + 2y = 0

Step 1: Find the characteristic equation.
The characteristic equation is r² - 3r + 2 = 0

Step 2: Factor the characteristic equation.
r² - 3r + 2 = (r - 1)(r - 2) = 0

Step 3: Find the roots.
r₁ = 1 and r₂ = 2

Step 4: Write the general solution.
Since we have two distinct real roots, the general solution is:
y = C₁e^(r₁x) + C₂e^(r₂x)
y = C₁e^x + C₂e^(2x)

Therefore, the general solution is:
y = C1*e^x + C2*e^(2x)"""
    },
    {
        "user": "Solve: y' = y/x",
        "assistant": """I'll solve this separable differential equation: y' = y/x

Step 1: Rearrange to separate variables.
dy/dx = y/x
dy/y = dx/x

Step 2: Integrate both sides.
∫(1/y)dy = ∫(1/x)dx
ln|y| = ln|x| + C₁

Step 3: Solve for y.
ln|y| = ln|x| + C₁
ln|y| = ln|x| + ln(C)  where C = e^(C₁)
ln|y| = ln|Cx|
y = Cx

Therefore, the general solution is:
y = C1*x

Note: This solution is valid for x ≠ 0."""
    }
]

def format_prompt(ode_text: str, system_prompt: Optional[str] = None, examples: Optional[List[Dict[str, str]]] = None) -> str:
    """Format the prompt for the LLM with system prompt, few-shot examples, and the ODE to solve."""
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    if examples is None:
        examples = FEW_SHOT_EXAMPLES
    
    # Construct the full prompt
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    for example in examples:
        prompt += f"<|im_start|>user\n{example['user']}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{example['assistant']}<|im_end|>\n"
    
    # Add the ODE to solve
    prompt += f"<|im_start|>user\nSolve the differential equation: {ode_text}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    return prompt

def extract_solution(generated_text: str) -> Optional[str]:
    """Extract the final symbolic solution from the generated text."""
    patterns = [
        r"y\s*=\s*([^\.]+)(?:\.|\n|$)",             
        r"y\(x\)\s*=\s*([^\.]+)(?:\.|\n|$)",        
        r"general solution is:?\s*y\s*=\s*([^\.]+)", 
        r"general solution is:?\s*y\(x\)\s*=\s*([^\.]+)", 
    ]
    
    for pattern in patterns:
        match = re.search(pattern, generated_text, re.DOTALL)
        if match:
            solution_text = match.group(1).strip()
            return solution_text
    
    return None

def symbolic_verification(ode_sympy: str, solution_sympy: str) -> Tuple[bool, Optional[float]]:
    """Verify if the solution satisfies the ODE using symbolic computation."""
    try:
        ode_eq = sp.sympify(ode_sympy)
        solution_eq = sp.sympify(solution_sympy)
        
        y_func = Function('y')(x)
        if isinstance(solution_eq, Eq):
            y_expr = solution_eq.rhs
        else:
            y_expr = solution_eq
        
        ode_subs = ode_eq.lhs.subs(y_func, y_expr) - ode_eq.rhs
        
        result = simplify(ode_subs)
        
        if result.is_constant():
            error = float(abs(result))
            is_valid = error < 1e-10
        else:
            error = 0.0
            for val in [0.1, 1.0, 2.0, 3.0, 4.0]:
                try:
                    error_at_point = abs(float(result.subs(x, val)))
                    error = max(error, error_at_point)
                except:
                    pass
            is_valid = error < 1e-6
        
        return is_valid, error
    except Exception as e:
        print(f"Error in symbolic verification: {e}")
        return False, None

def load_model(model_name: str, use_half_precision: bool = True, device: str = "auto"):
    """Load the DeepSeek-R1 model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_half_precision and device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device
        )
    
    return model, tokenizer

def solve_ode(ode_text: str, model, tokenizer, temperature: float = 0.1, max_tokens: int = 1024) -> Dict[str, Any]:
    """Solve the ODE using the LLM and parse the response."""
    start_time = time.time()
    
    prompt = format_prompt(ode_text)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=False)
    
    assistant_response = generated_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    solution_text = extract_solution(assistant_response)
    
    end_time = time.time()
    
    result = {
        "ode_text": ode_text,
        "full_response": assistant_response,
        "solution_text": solution_text,
        "time_taken": end_time - start_time,
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Solve ODEs using DeepSeek-R1")
    parser.add_argument("--ode", type=str, help="ODE to solve in plain text")
    parser.add_argument("--file", type=str, help="JSON file containing ODEs to solve")
    parser.add_argument("--output", type=str, default="solutions.json", help="Output file for solutions")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-r1-7b-base", help="Model name or path")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    args = parser.parse_args()
    
    if not args.ode and not args.file:
        parser.error("At least one of --ode or --file must be provided")
    
    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model, use_half_precision=args.half, device=args.device)
    print("Model loaded successfully")
    
    results = []
    
    if args.ode:
        print(f"Solving ODE: {args.ode}")
        result = solve_ode(args.ode, model, tokenizer, args.temperature, args.max_tokens)
        print(f"Solution: {result['solution_text']}")
        results.append(result)
    
    if args.file:
        print(f"Solving ODEs from file {args.file}")
        with open(args.file, 'r') as f:
            odes = json.load(f)
        
        for i, ode_data in enumerate(odes):
            ode_text = ode_data["ode_text"]
            print(f"[{i+1}/{len(odes)}] Solving ODE: {ode_text}")
            
            result = solve_ode(ode_text, model, tokenizer, args.temperature, args.max_tokens)
            
            if "solution_sympy" in ode_data:
                result["reference_solution"] = ode_data["solution_sympy"]
                
                if result["solution_text"] and result["reference_solution"]:
                    try:
                        model_sol = result["solution_text"].replace("C1", "C1").replace("C2", "C2")
                        model_sol = f"Eq(y(x), {model_sol})"
                        
                        is_valid, error = symbolic_verification(ode_data["ode_sympy"], model_sol)
                        result["verification"] = {
                            "is_valid": is_valid,
                            "error": error
                        }
                    except Exception as e:
                        result["verification"] = {
                            "is_valid": False,
                            "error": str(e)
                        }
            
            results.append(result)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    if len(results) > 1:
        solved_count = sum(1 for r in results if r.get("solution_text") is not None)
        verified_count = sum(1 for r in results if r.get("verification", {}).get("is_valid", False))
        
        print("\nSummary:")
        print(f"Total ODEs: {len(results)}")
        print(f"Solutions extracted: {solved_count} ({solved_count/len(results)*100:.1f}%)")
        
        if verified_count > 0:
            print(f"Verified correct: {verified_count} ({verified_count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main() 