# config.py
"""
Configuration dataclasses for PIBLS model and training parameters.

This module defines structured configurations for:
- ModelConfig: BLS network architecture parameters
- TrainingConfig: Training loop and data generation settings

PDE-related dataclasses (PDETerm, NonlinearTerm, ResidualTerm) are defined
in physics/baseproblem.py as they are physics domain concepts.
"""
from dataclasses import dataclass, field
from typing import Union, List, Callable, Optional

import torch

@dataclass
class ModelConfig:
    """
    Model hyperparameter configuration for the Broad Learning System.

    Attributes:
        num_feature: Number of feature (mapping) nodes in the BLS
        num_enhancement: Number of enhancement nodes in the BLS
        feature_activation: Activation function for feature layer ('tanh', 'sin', 'sigmoid', 'relu')
        enhancement_activation: Activation function for enhancement layer
        feature_hidden_dim: Hidden dimension for feature nodes ('auto' uses input dimension)
        enhancement_hidden_dim: Hidden dimension for enhancement nodes ('auto' uses feature output dimension)
        dtype: PyTorch data type for computations
        bls_random_range: Range for random weight initialization [-range, range]
        bls_enhance_whiten: Whether to orthogonalize enhancement layer weights
    """
    num_feature: int = 10
    num_enhancement: int = 10
    feature_activation: str = "tanh"
    enhancement_activation: str = "tanh"
    feature_hidden_dim: Union[int, str] = "auto"
    enhancement_hidden_dim: Union[int, str] = "auto"
    dtype: torch.dtype = torch.float64
    # BLS random range
    bls_random_range: float = 5.0
    bls_enhance_whiten: bool = False

@dataclass
class TrainingConfig:
    """
    Training and data generation configuration.

    Attributes:
        n_colloc: Number of collocation points for PDE residual
        n_bc: Number of boundary condition points
        n_init: Number of grid-based initial condition points
        n_ic_data: Number of external IC data points to load from file
        data_root: Root directory for data files
        normalize_input: Whether to normalize inputs to [-1, 1]
        delta: Perturbation magnitude for perturbed least squares
        cost_threshold: Convergence threshold for nonlinear solver
        max_sub_iterations: Maximum perturbation iterations for nonlinear solver
        seed: Random seed for reproducibility
        results_dir: Directory to save results
        results_name: Filename for experiment results CSV
        dtype: PyTorch data type for computations
        points_verbose: Whether to visualize point distributions
        visualize: Whether to generate solution plots after training
        save_model: Whether to save model after training
    """
    n_colloc: int = 200
    n_bc: int = 2
    n_init: int = 0

    # External IC data configuration (for Fisher problems)
    n_ic_data: int = 0
    data_root: str = "./data"

    # Training
    normalize_input: bool = True
    delta: float = 0.5
    cost_threshold: float = 1e-3
    max_sub_iterations: int = 5
    seed: int = 3407
    dtype: torch.dtype = torch.float64

    # Output configuration
    results_dir: str = "./results"
    results_name: str = "experiment_results.csv"
    points_verbose: bool = False  # Whether to visualize point distributions
    visualize: bool = True  # Whether to visualize solution after training
    save_model: bool = True  # Whether to save model after training

    # Evaluation configuration
    n_eval: int = 0  # Number of evaluation points (0 = auto-compute from training points)
    eval_multiplier: int = 10  # Multiplier for auto-computing evaluation points
    eval_max_points: int = 5000  # Maximum evaluation points per dimension