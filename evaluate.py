#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sympy as sp
from sympy import symbols, Function, dsolve, simplify, Eq, sympify
from sympy.abc import x, C1, C2

def load_solution_results(results_file: str):
    """Load solution results from a JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def algebraically_equivalent(expr1_str: str, expr2_str: str) -> bool:
    try:
        expr1 = sympify(expr1_str)
        expr2 = sympify(expr2_str)
        
        diff = simplify(expr1 - expr2)
        
        if diff == 0:
            return True
        
        c1_var = sp.Symbol('c1_var')
        c2_var = sp.Symbol('c2_var')
        
        diff_with_vars = diff.subs({C1: c1_var, C2: c2_var})
        
        if diff_with_vars == 0:
            return True
        
        return False
    except Exception as e:
        print(f"Error comparing expressions: {e}")
        return False

def verify_solution(ode_eq_str: str, correct_solution_str: str, model_solution_str: str) -> bool:
    """
    Verify if the model's solution is correct by:
    1. Checking if solutions are algebraically equivalent
    2. Checking if the model's solution satisfies the ODE
    """
    try:
        if algebraically_equivalent(correct_solution_str, model_solution_str):
            return True
        
        ode_eq = sympify(ode_eq_str)
        
        model_solution = sympify(model_solution_str)
        
        y_func = Function('y')(x)
        
        ode_subs = ode_eq.lhs.subs(y_func, model_solution) - ode_eq.rhs
        
        result = simplify(ode_subs)
        
        if result == 0:
            return True
        
        is_valid = True
        for val in [0.5, 1.0, 2.0]:
            try:
                error = abs(float(result.subs(x, val)))
                if error > 1e-6:
                    is_valid = False
                    break
            except:
                pass
        
        return is_valid
    except Exception as e:
        print(f"Error verifying solution: {e}")
        return False

def evaluate_solutions(results, dataset_file=None):
    """
    Evaluate the solutions by comparing with reference solutions
    or verifying if they satisfy the original ODEs.
    """
    if dataset_file:
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        ode_map = {item["ode_text"]: item for item in dataset}
        
        for result in results:
            ode_text = result["ode_text"]
            if ode_text in ode_map:
                result["ode_sympy"] = ode_map[ode_text]["ode_sympy"]
                result["reference_solution"] = ode_map[ode_text]["solution_sympy"]
    
    for result in tqdm(results, desc="Evaluating solutions"):
        if "solution_text" not in result or not result["solution_text"]:
            result["evaluation"] = {
                "success": False,
                "reason": "No solution extracted"
            }
            continue
        
        if "reference_solution" in result:
            try:
                model_sol = result["solution_text"]
                
                ref_sol = result["reference_solution"]
                if isinstance(ref_sol, str) and "Eq(" in ref_sol:
                    ref_expr = ref_sol.split(",", 1)[1].strip().rstrip(")")
                else:
                    ref_expr = ref_sol
                
                is_correct = verify_solution(
                    result["ode_sympy"],
                    ref_expr,
                    model_sol
                )
                
                result["evaluation"] = {
                    "success": is_correct,
                    "reason": "Solution is correct" if is_correct else "Solution is incorrect"
                }
            except Exception as e:
                result["evaluation"] = {
                    "success": False,
                    "reason": f"Error during verification: {str(e)}"
                }
        else:
            result["evaluation"] = {
                "success": True if result["solution_text"] else False,
                "reason": "Solution extracted, but couldn't verify correctness"
            }
    
    return results

def compute_metrics(evaluated_results):
    """Compute performance metrics from evaluated results."""
    total = len(evaluated_results)
    solution_extracted = sum(1 for r in evaluated_results if r.get("solution_text"))
    solution_correct = sum(1 for r in evaluated_results 
                          if r.get("evaluation", {}).get("success", False))
    
    types_stats = {}
    for result in evaluated_results:
        if "ode_type" in result:
            ode_type = result["ode_type"]
            if ode_type not in types_stats:
                types_stats[ode_type] = {"total": 0, "extracted": 0, "correct": 0}
            
            types_stats[ode_type]["total"] += 1
            if result.get("solution_text"):
                types_stats[ode_type]["extracted"] += 1
            if result.get("evaluation", {}).get("success", False):
                types_stats[ode_type]["correct"] += 1
    
    times = [r.get("time_taken", 0) for r in evaluated_results if "time_taken" in r]
    avg_time = np.mean(times) if times else 0
    
    metrics = {
        "total": total,
        "solution_extracted": solution_extracted,
        "solution_extracted_percentage": solution_extracted / total * 100 if total else 0,
        "solution_correct": solution_correct,
        "solution_correct_percentage": solution_correct / total * 100 if total else 0,
        "average_time": avg_time,
        "types_stats": types_stats
    }
    
    return metrics

def plot_results(metrics, output_file="evaluation_results.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = ['Total', 'Solution Extracted', 'Solution Correct']
    values = [metrics["total"], metrics["solution_extracted"], metrics["solution_correct"]]
    
    ax1.bar(labels, values, color=['blue', 'green', 'orange'])
    ax1.set_title('Overall Performance')
    ax1.set_ylabel('Count')
    
    for i, v in enumerate(values):
        if i > 0:
            percentage = v / values[0] * 100 if values[0] > 0 else 0
            ax1.text(i, v + 5, f"{percentage:.1f}%", ha='center')
    
    if metrics["types_stats"]:
        types = list(metrics["types_stats"].keys())
        correct_rates = [stats["correct"] / stats["total"] * 100 
                         for type_name, stats in metrics["types_stats"].items()]
        
        ax2.bar(types, correct_rates, color='purple')
        ax2.set_title('Correctness by ODE Type')
        ax2.set_ylabel('Correct Solutions (%)')
        ax2.set_ylim(0, 100)
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results plot saved to {output_file}")

def generate_report(metrics, evaluated_results, output_file="evaluation_report.md"):
    """Generate a detailed markdown report of the evaluation results."""
    report_lines = [
        "# ODE Solver Evaluation Report",
        "",
        "## Summary",
        f"- **Total ODEs evaluated**: {metrics['total']}",
        f"- **Solutions extracted**: {metrics['solution_extracted']} ({metrics['solution_extracted_percentage']:.1f}%)",
        f"- **Correct solutions**: {metrics['solution_correct']} ({metrics['solution_correct_percentage']:.1f}%)",
        f"- **Average solution time**: {metrics['average_time']:.2f} seconds",
        ""
    ]
    
    if metrics["types_stats"]:
        report_lines.extend([
            "## Performance by ODE Type",
            "",
            "| ODE Type | Total | Solutions Extracted | Correct Solutions | Success Rate |",
            "| -------- | ----- | ------------------- | ----------------- | ------------ |"
        ])
        
        for ode_type, stats in metrics["types_stats"].items():
            success_rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            report_lines.append(
                f"| {ode_type} | {stats['total']} | {stats['extracted']} ({stats['extracted']/stats['total']*100:.1f}%) | "
                f"{stats['correct']} | {success_rate:.1f}% |"
            )
        
        report_lines.append("")
    
    successes = [r for r in evaluated_results if r.get("evaluation", {}).get("success", False)]
    failures = [r for r in evaluated_results if not r.get("evaluation", {}).get("success", False)]
    
    report_lines.extend([
        "## Example Successes",
        ""
    ])
    
    for i, success in enumerate(successes[:3]):  
        report_lines.extend([
            f"### Success Example {i+1}",
            f"**ODE**: `{success['ode_text']}`",
            "",
            "**Model Solution**:",
            f"```",
            f"{success.get('solution_text', 'N/A')}",
            f"```",
            "",
            "**Reference Solution**:",
            f"```",
            f"{success.get('reference_solution', 'N/A')}",
            f"```",
            ""
        ])
    
    report_lines.extend([
        "## Example Failures",
        ""
    ])
    
    for i, failure in enumerate(failures[:3]):
        report_lines.extend([
            f"### Failure Example {i+1}",
            f"**ODE**: `{failure['ode_text']}`",
            "",
            "**Model Solution**:",
            f"```",
            f"{failure.get('solution_text', 'N/A')}",
            f"```",
            "",
            "**Reference Solution**:",
            f"```",
            f"{failure.get('reference_solution', 'N/A')}",
            f"```",
            "",
            f"**Reason**: {failure.get('evaluation', {}).get('reason', 'Unknown')}",
            ""
        ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Evaluation report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate ODE solver performance")
    parser.add_argument("--results", type=str, required=True, help="JSON file with solver results")
    parser.add_argument("--dataset", type=str, help="Original dataset file for reference solutions")
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Directory to save evaluation results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {args.results}")
    results = load_solution_results(args.results)
    print(f"Loaded {len(results)} results")
    
    print("Evaluating solutions...")
    evaluated_results = evaluate_solutions(results, args.dataset)
    
    print("Computing metrics...")
    metrics = compute_metrics(evaluated_results)
    
    print("Generating plots...")
    plot_results(metrics, output_file=output_dir / "evaluation_results.png")
    
    print("Generating report...")
    generate_report(metrics, evaluated_results, output_file=output_dir / "evaluation_report.md")
    
    with open(output_dir / "evaluated_results.json", 'w') as f:
        json.dump(evaluated_results, f, indent=2)
    
    print("\nEvaluation Summary:")
    print(f"Total ODEs: {metrics['total']}")
    print(f"Solutions extracted: {metrics['solution_extracted']} ({metrics['solution_extracted_percentage']:.1f}%)")
    print(f"Correct solutions: {metrics['solution_correct']} ({metrics['solution_correct_percentage']:.1f}%)")
    print(f"Average solution time: {metrics['average_time']:.2f} seconds")
    
    print(f"\nDetailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 