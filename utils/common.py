"""
Common utilities for PIBLS.

This module provides general-purpose utilities used across the codebase:
- Random seed management
- Error metrics calculation
- Experiment result logging
- Plot data saving
- Nonlinear least squares optimization
"""
import os
import csv
import time
import random
from dataclasses import asdict
from typing import Callable, Dict, Any

import torch
import numpy as np
from scipy.optimize import least_squares


def set_random_seeds(seed=3407):
    """
    Set random seeds for reproducibility across all random number generators.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_errors(u_pred: torch.Tensor, u_exact: torch.Tensor):
    """
    Calculate Max, Mean, and L2 errors between predicted and exact solutions.

    Args:
        u_pred: Predicted solution tensor of shape (n_points, 1)
        u_exact: Exact solution tensor of shape (n_points, 1)

    Returns:
        tuple: (max_error, mean_error, l2_error, error_tensor)
            - max_error: Maximum absolute error (L-infinity norm)
            - mean_error: Mean absolute error (L1 norm normalized)
            - l2_error: Root mean square error (L2 norm normalized)
            - error_tensor: Full error tensor (u_pred - u_exact)
    """
    error = u_pred - u_exact
    max_error = torch.max(torch.abs(error)).item()
    mean_error = torch.mean(torch.abs(error)).item()
    l2_error = torch.sqrt(torch.mean(error**2)).item()

    return max_error, mean_error, l2_error, error


def log_results(save_path, model_cfg, train_cfg, problem_name, metrics, elapsed_time):
    """
    Log experiment results to a CSV file.

    Appends a new row to the CSV file (creates if not exists).

    Args:
        save_path: Path to the CSV file
        model_cfg: ModelConfig dataclass instance
        train_cfg: TrainingConfig dataclass instance
        problem_name: Name of the problem being solved
        metrics: Tuple of (max_error, mean_error, l2_error)
        elapsed_time: Training time in seconds
    """
    # Prepare data dictionaries
    model_dict = {f"model_{k}": v for k, v in asdict(model_cfg).items()}
    train_dict = {f"train_{k}": v for k, v in asdict(train_cfg).items()}

    # Convert dtype to string representation
    if 'model_dtype' in model_dict:
        model_dict['model_dtype'] = str(model_dict['model_dtype'])
    if 'train_dtype' in train_dict:
        train_dict['train_dtype'] = str(train_dict['train_dtype'])

    results_dict = {
        "Problem": problem_name,
        "Max_Error": f"{metrics[0]:.4e}",
        "Mean_Error": f"{metrics[1]:.4e}",
        "L2_Error": f"{metrics[2]:.4e}",
        "Training_Time_sec": f"{elapsed_time:.2f}"
    }

    log_data = {**results_dict, **model_dict, **train_dict}

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.isfile(save_path)

    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    print(f"Results logged to {save_path}")


def save_plot_data(save_dir: str, filename: str, **data: Dict[str, Any]):
    """
    Save plot data to an npz file for reproducibility.

    Args:
        save_dir: Directory to save the npz file (will create if not exists)
        filename: Name of the npz file (with or without .npz extension)
        **data: Keyword arguments of data arrays to save
                Tensors will be converted to numpy arrays automatically
    """
    os.makedirs(save_dir, exist_ok=True)

    # Ensure .npz extension
    if not filename.endswith('.npz'):
        filename = f"{filename}.npz"

    save_path = os.path.join(save_dir, filename)

    # Convert tensors to numpy
    numpy_data = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            numpy_data[key] = value.detach().cpu().numpy()
        else:
            numpy_data[key] = value

    np.savez(save_path, **numpy_data)
    return save_path


def nlsq_perturb(func: Callable, jac: Callable, x0: np.ndarray, delta: float,
                 cost_threshold: float, max_sub_iterations: int,
                 verbose: int = 1, **kwargs):
    """
    Perturbed nonlinear least squares solver.

    This solver addresses local minima issues in nonlinear optimization by:
    1. Running initial optimization from x0
    2. If cost > threshold, perturbing the best solution and re-optimizing
    3. Keeping track of the best solution found across all perturbations

    Args:
        func: Residual function f(x) -> residual vector
        jac: Jacobian function J(x) -> Jacobian matrix
        x0: Initial guess for parameters
        delta: Maximum perturbation magnitude
        cost_threshold: Stop if cost falls below this value
        max_sub_iterations: Maximum number of perturbation attempts
        verbose: Verbosity level (0=silent, 1=summary, 2=detailed)
        **kwargs: Additional arguments passed to scipy.optimize.least_squares

    Returns:
        OptimizeResult with best solution found
    """
    in_time = time.time()
    x0 = x0.flatten()

    if verbose >= 1:
        print(f"Starting initial least_squares optimization...")

    initial_start = time.time()
    res = least_squares(func, x0, jac=jac, **kwargs)
    initial_time = time.time() - initial_start

    if verbose >= 1:
        print(f"Initial least_squares took: {initial_time:.4f} seconds, cost = {res.cost:.4e}")

    x = res.x
    c = res.cost

    if c < cost_threshold:
        if verbose >= 1:
            print(f"Initial cost {c:.4e} is below threshold. Total time: {time.time() - in_time:.4f}s")
        return res

    best_x = np.copy(x)
    best_cost = c
    best_res = res

    # Perturbation iterations to escape local minima
    for i in range(max_sub_iterations):
        iter_start = time.time()

        # Generate random perturbation
        xi1 = np.random.rand()
        delta1 = xi1 * delta
        delta_x = np.random.uniform(-delta1, delta1, size=x.shape)
        xi2 = np.random.rand()
        y0_perturbed = xi2 * best_x + delta_x

        try:
            perturb_start = time.time()
            res_perturb = least_squares(func, y0_perturbed, jac=jac, **kwargs)
            perturb_time = time.time() - perturb_start

            cost_perturb = res_perturb.cost
            x_perturb = res_perturb.x

            if verbose >= 2:
                print(f"Sub-iteration {i+1}: least_squares took {perturb_time:.4f}s, cost = {cost_perturb:.4e}")

            if cost_perturb < best_cost:
                best_x = x_perturb
                best_cost = cost_perturb
                best_res = res_perturb
                if verbose >= 2:
                    print(f"  -> New best solution found!")

            if best_cost < cost_threshold:
                total_time = time.time() - in_time
                if verbose >= 1:
                    print(f"Converged in {i+1} iterations. Total time: {total_time:.4f}s")
                best_res.x = best_x
                best_res.cost = best_cost
                return best_res

        except Exception as e:
            if verbose >= 2:
                print(f"  Error in sub-iteration {i+1}: {e}")
            continue

        if verbose >= 2:
            iter_time = time.time() - iter_start
            print(f"Sub-iteration {i+1} total time: {iter_time:.4f}s")

    total_time = time.time() - in_time
    if verbose >= 1:
        print(f"Max iterations ({max_sub_iterations}) reached. Best cost: {best_cost:.4e}, Total time: {total_time:.4f}s")

    best_res.x = best_x
    best_res.cost = best_cost
    return best_res
