#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import random
import sympy as sp
from sympy.abc import x, y, C1, C2
from sympy import Function, dsolve, Eq, exp, sin, cos, log, sqrt
from tqdm import tqdm

def generate_separable_ode():
    """Generate a separable ODE of the form g(y) * dy/dx = f(x)."""
    g_options = [
        (y, "y"),
        (y**2, "y^2"),
        (y**3, "y^3"),
        (sin(y), "sin(y)"),
        (exp(y), "e^y"),
        (1/y, "1/y"),
        (sqrt(y), "sqrt(y)"),
    ]
    
    f_options = [
        (x, "x"),
        (x**2, "x^2"),
        (sin(x), "sin(x)"),
        (exp(x), "e^x"),
        (1/x, "1/x"),
        (sqrt(x), "sqrt(x)"),
    ]
    
    g_expr, g_text = random.choice(g_options)
    f_expr, f_text = random.choice(f_options)
    
    y_func = Function('y')(x)
    ode = Eq(g_expr * y_func.diff(x), f_expr)
    
    text_repr = f"{g_text} * dy/dx = {f_text}"
    
    try:
        solution = dsolve(ode, y_func)
        return {
            "ode_type": "separable",
            "ode_sympy": str(ode),
            "ode_text": text_repr,
            "solution_sympy": str(solution),
            "solution_text": str(solution).replace("y(x)", "y")
        }
    except Exception as e:
        return generate_separable_ode()

def generate_linear_first_order():
    """Generate a linear first-order ODE of the form dy/dx + P(x)y = Q(x)."""
    p_options = [
        (1, "1"),
        (x, "x"),
        (sin(x), "sin(x)"),
        (1/x, "1/x"),
        (2*x, "2x"),
    ]
    
    q_options = [
        (x, "x"),
        (x**2, "x^2"),
        (sin(x), "sin(x)"),
        (exp(x), "e^x"),
        (x*exp(x), "x*e^x"),
    ]
    
    p_expr, p_text = random.choice(p_options)
    q_expr, q_text = random.choice(q_options)
    
    y_func = Function('y')(x)
    ode = Eq(y_func.diff(x) + p_expr * y_func, q_expr)
    
    text_repr = f"dy/dx + {p_text}*y = {q_text}"
    
    try:
        solution = dsolve(ode, y_func)
        return {
            "ode_type": "linear_first_order",
            "ode_sympy": str(ode),
            "ode_text": text_repr,
            "solution_sympy": str(solution),
            "solution_text": str(solution).replace("y(x)", "y")
        }
    except Exception as e:
        return generate_linear_first_order()

def generate_bernoulli():
    """Generate a Bernoulli ODE of the form dy/dx + P(x)y = Q(x)y^n."""
    p_options = [
        (1, "1"),
        (x, "x"),
        (sin(x), "sin(x)"),
    ]
    
    q_options = [
        (1, "1"),
        (x, "x"),
        (x**2, "x^2"),
        (exp(x), "e^x"),
    ]
    
    n_options = [2, 3, -1, -2]
    
    p_expr, p_text = random.choice(p_options)
    q_expr, q_text = random.choice(q_options)
    n = random.choice(n_options)
    
    y_func = Function('y')(x)
    ode = Eq(y_func.diff(x) + p_expr * y_func, q_expr * y_func**n)
    
    text_repr = f"dy/dx + {p_text}*y = {q_text}*y^{n}"
    
    try:
        solution = dsolve(ode, y_func)
        return {
            "ode_type": "bernoulli",
            "ode_sympy": str(ode),
            "ode_text": text_repr,
            "solution_sympy": str(solution),
            "solution_text": str(solution).replace("y(x)", "y")
        }
    except Exception as e:
        return generate_bernoulli()

def generate_second_order_constant_coeff():
    """Generate a second-order ODE with constant coefficients: a*y'' + b*y' + c*y = f(x)."""
    # Define coefficients a, b, c
    a = random.choice([1, 2, 3])
    b = random.choice([-3, -2, -1, 0, 1, 2, 3])
    c = random.choice([-3, -2, -1, 0, 1, 2, 3])
    
    f_options = [
        (0, "0"),
        (x, "x"),
        (x**2, "x^2"),
        (sin(x), "sin(x)"),
        (cos(x), "cos(x)"),
        (exp(x), "e^x"),
    ]
    
    f_expr, f_text = random.choice(f_options)
    
    y_func = Function('y')(x)
    ode = Eq(a * y_func.diff(x, 2) + b * y_func.diff(x) + c * y_func, f_expr)
    
    if a == 1:
        a_text = ""
    else:
        a_text = str(a)
    
    text_repr = f"{a_text}y'' "
    
    if b != 0:
        b_sign = "+" if b > 0 else "-"
        text_repr += f"{b_sign} {abs(b)}y' "
    
    if c != 0:
        c_sign = "+" if c > 0 else "-"
        text_repr += f"{c_sign} {abs(c)}y "
    
    text_repr += f"= {f_text}"
    text_repr = text_repr.strip()
    
    try:
        solution = dsolve(ode, y_func)
        return {
            "ode_type": "second_order_constant_coeff",
            "ode_sympy": str(ode),
            "ode_text": text_repr,
            "solution_sympy": str(solution),
            "solution_text": str(solution).replace("y(x)", "y")
        }
    except Exception as e:
        return generate_second_order_constant_coeff()

def generate_cauchy_euler():
    """Generate a Cauchy-Euler equation of the form x^2*y'' + a*x*y' + b*y = 0."""
    a = random.choice([-2, -1, 0, 1, 2, 3, 4])
    b = random.choice([-2, -1, 0, 1, 2, 3, 4])
    
    y_func = Function('y')(x)
    ode = Eq(x**2 * y_func.diff(x, 2) + a * x * y_func.diff(x) + b * y_func, 0)
    
    text_repr = f"x^2*y'' "
    
    if a != 0:
        a_sign = "+" if a > 0 else "-"
        text_repr += f"{a_sign} {abs(a)}x*y' "
    
    if b != 0:
        b_sign = "+" if b > 0 else "-"
        text_repr += f"{b_sign} {abs(b)}y "
    
    text_repr += "= 0"
    text_repr = text_repr.strip()
    
    try:
        solution = dsolve(ode, y_func)
        return {
            "ode_type": "cauchy_euler",
            "ode_sympy": str(ode),
            "ode_text": text_repr,
            "solution_sympy": str(solution),
            "solution_text": str(solution).replace("y(x)", "y")
        }
    except Exception as e:
        return generate_cauchy_euler()

def generate_ode_dataset(size, seed=42):
    """Generate a dataset of ODEs with solutions."""
    random.seed(seed)
    dataset = []
    
    generators = [
        generate_separable_ode,
        generate_linear_first_order,
        generate_bernoulli,
        generate_second_order_constant_coeff,
        generate_cauchy_euler
    ]
    
    for _ in tqdm(range(size), desc="Generating ODEs"):
        generator = random.choice(generators)
        ode_data = generator()
        dataset.append(ode_data)
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Generate ODE datasets")
    parser.add_argument("--train_size", type=int, default=300, help="Size of training dataset")
    parser.add_argument("--eval_size", type=int, default=100, help="Size of evaluation dataset")
    parser.add_argument("--train_output", type=str, default="ode_train.json", help="Path to save training dataset")
    parser.add_argument("--eval_output", type=str, default="ode_eval.json", help="Path to save evaluation dataset")
    parser.add_argument("--train_seed", type=int, default=42, help="Random seed for training data generation")
    parser.add_argument("--eval_seed", type=int, default=43, help="Random seed for evaluation data generation")
    args = parser.parse_args()
    
    print(f"Generating training dataset ({args.train_size} examples)...")
    train_dataset = generate_ode_dataset(args.train_size, seed=args.train_seed)
    
    print(f"Generating evaluation dataset ({args.eval_size} examples)...")
    eval_dataset = generate_ode_dataset(args.eval_size, seed=args.eval_seed)
    
    with open(args.train_output, 'w') as f:
        json.dump(train_dataset, f, indent=2)
    
    with open(args.eval_output, 'w') as f:
        json.dump(eval_dataset, f, indent=2)
    
    print(f"Training dataset saved to {args.train_output}")
    print(f"Evaluation dataset saved to {args.eval_output}")
    
    train_types = {}
    for item in train_dataset:
        ode_type = item["ode_type"]
        train_types[ode_type] = train_types.get(ode_type, 0) + 1
    
    eval_types = {}
    for item in eval_dataset:
        ode_type = item["ode_type"]
        eval_types[ode_type] = eval_types.get(ode_type, 0) + 1
    
    print("\nTraining dataset statistics:")
    for ode_type, count in train_types.items():
        print(f"  {ode_type}: {count} ({count/len(train_dataset)*100:.1f}%)")
    
    print("\nEvaluation dataset statistics:")
    for ode_type, count in eval_types.items():
        print(f"  {ode_type}: {count} ({count/len(eval_dataset)*100:.1f}%)")

if __name__ == "__main__":
    main() 