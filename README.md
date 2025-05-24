# DeepSeek-R1 ODE Solver

Leveraging the DeepSeek-R1 Open-Source Large Language Model for Automated Solution of Ordinary Differential Equations (ODEs).

## Overview

This project explores using DeepSeek-R1, a 7B parameter open-source LLM, to solve ordinary differential equations expressed in natural language. The system accepts ODEs in plain text and returns symbolic general solutions, which are then verified against SymPy's dsolve outputs.

## Features

- Dataset generation for training and evaluation
- Prompt-engineered ODE solving pipeline
- Automatic verification of solutions
- (Optional) LoRA fine-tuning capabilities

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd deepseek-ode-solver

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generating ODE Datasets

```bash
python dataset_builder.py --train_size 300 --eval_size 100
```

### Running the ODE Solver

```bash
python solver.py --ode "y'(x) + y(x) = x^2" --model "deepseek-ai/deepseek-r1-7b-base"
```

### Evaluating Solver Performance

```bash
python evaluate.py --dataset "ode_eval.json" --model "deepseek-ai/deepseek-r1-7b-base"
```

### Fine-tuning (Optional)

```bash
python finetune_lora.py --train_dataset "ode_train.json" --eval_dataset "ode_eval.json"
```

## Components

- `dataset_builder.py`: Generates train and evaluation ODE datasets
- `solver.py`: Main module for solving ODEs using the LLM
- `evaluate.py`: Evaluates the performance of the solver
- `finetune_lora.py`: (Optional) Fine-tunes the model using LoRA
- `ode_train.json`: Training dataset of ODE problems and solutions
- `ode_eval.json`: Evaluation dataset of ODE problems and solutions

## License

MIT

## References

1. DeepSeek-AI. *DeepSeek-R1: Toward Reasoning in Large Language Models*. 2024.
2. Aaron Meurer et al. *SymPy: Symbolic Computing in Python*. *PeerJ Computer Science* 2017.
3. Yuntian Li et al. *Transformers and Math Reasoning*. *arXiv* 2023. 