#!/usr/bin/env python3
"""
PIBLS - Physics-Informed Broad Learning System for solving PDEs.

Usage:
    python main.py --problem TC1
    python main.py --config configs/tc1.yaml
    python main.py --model_path results/models/TC1.pt
    python main.py --list-problems
"""
import argparse

from physics.registry import get_problem_class, list_problems
from models.pibls import PIBLS
from trainer.solver import PIBLSSolver
from utils.config_loader import load_config, parse_cli_overrides, dict_to_model_config, dict_to_training_config


def main():
    parser = argparse.ArgumentParser(description="PIBLS - Physics-Informed Broad Learning System")

    parser.add_argument("--problem", "-p", type=str, help="Problem to solve")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")
    parser.add_argument("--model_path", "-m", type=str, help="Path to saved model (skip training if config matches)")
    parser.add_argument("--list-problems", action="store_true", help="List available problems")

    # Config overrides
    parser.add_argument("--n_colloc", type=int)
    parser.add_argument("--n_bc", type=int)
    parser.add_argument("--n_init", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_feature", type=int)
    parser.add_argument("--num_enhancement", type=int)
    parser.add_argument("--visualize", action="store_true", default=None)
    parser.add_argument("--no-visualize", action="store_false", dest="visualize")
    parser.add_argument("--save_model", action="store_true", default=None)
    parser.add_argument("--results_dir", type=str)

    args = parser.parse_args()

    if args.list_problems:
        print("\nAvailable problems:")
        for name in list_problems():
            print(f"  {name}")
        return

    config = load_config(args.config, parse_cli_overrides(args))
    model_cfg = dict_to_model_config(config['model'])
    train_cfg = dict_to_training_config(config['training'])

    problem = get_problem_class(config['problem'])()
    model = PIBLS(model_cfg)

    solver = PIBLSSolver(problem, model, model_cfg, train_cfg, model_path=args.model_path)
    solver.train()
    solver.evaluate()
    if train_cfg.visualize:
        solver.visualize()


if __name__ == "__main__":
    main()
