"""
YAML configuration loading utilities.

This module provides functions for:
- Loading YAML configuration files
- Merging configurations (base + overrides)
- Converting between YAML and dataclasses
- Saving resolved configurations for reproducibility
"""
import yaml
import json
from pathlib import Path
from dataclasses import asdict, fields
from typing import Dict, Any, Optional, Type, TypeVar
from copy import deepcopy

import torch

from configs.config import ModelConfig, TrainingConfig


T = TypeVar('T')


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return as dictionary.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary containing YAML contents

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """
    Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save dictionary to JSON file (human-readable format).

    Args:
        data: Dictionary to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def _convert_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'torch.float32': torch.float32,
        'torch.float64': torch.float64,
    }
    return dtype_map.get(dtype_str, torch.float64)


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to string."""
    return str(dtype).replace('torch.', '')


def dict_to_model_config(data: Dict[str, Any]) -> ModelConfig:
    """
    Convert dictionary to ModelConfig dataclass.

    Args:
        data: Dictionary with model configuration

    Returns:
        ModelConfig instance
    """
    # Handle dtype conversion
    if 'dtype' in data and isinstance(data['dtype'], str):
        data = data.copy()
        data['dtype'] = _convert_dtype(data['dtype'])

    # Filter to only valid fields
    valid_fields = {f.name for f in fields(ModelConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return ModelConfig(**filtered)


def dict_to_training_config(data: Dict[str, Any]) -> TrainingConfig:
    """
    Convert dictionary to TrainingConfig dataclass.

    Args:
        data: Dictionary with training configuration

    Returns:
        TrainingConfig instance
    """
    # Handle dtype conversion
    if 'dtype' in data and isinstance(data['dtype'], str):
        data = data.copy()
        data['dtype'] = _convert_dtype(data['dtype'])

    # Filter to only valid fields
    valid_fields = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return TrainingConfig(**filtered)


def model_config_to_dict(config: ModelConfig) -> Dict[str, Any]:
    """
    Convert ModelConfig to dictionary (YAML-serializable).

    Args:
        config: ModelConfig instance

    Returns:
        Dictionary representation
    """
    data = asdict(config)
    # Convert dtype to string
    if 'dtype' in data:
        data['dtype'] = _dtype_to_str(data['dtype'])
    return data


def training_config_to_dict(config: TrainingConfig) -> Dict[str, Any]:
    """
    Convert TrainingConfig to dictionary (YAML-serializable).

    Args:
        config: TrainingConfig instance

    Returns:
        Dictionary representation
    """
    data = asdict(config)
    # Convert dtype to string
    if 'dtype' in data:
        data['dtype'] = _dtype_to_str(data['dtype'])
    return data


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional CLI overrides.

    Precedence: CLI overrides > YAML config > defaults

    Args:
        config_path: Path to YAML config file (optional)
        cli_overrides: Dictionary of CLI override values (optional)

    Returns:
        Dictionary with merged configuration containing:
        - 'problem': Problem name string
        - 'model': Model configuration dict
        - 'training': Training configuration dict
    """
    # Start with defaults
    default_config = {
        'problem': 'TC1',
        'model': model_config_to_dict(ModelConfig()),
        'training': training_config_to_dict(TrainingConfig()),
    }

    # Load YAML if provided
    if config_path:
        yaml_config = load_yaml(config_path)
        config = deep_merge(default_config, yaml_config)
    else:
        config = default_config

    # Apply CLI overrides
    if cli_overrides:
        config = deep_merge(config, cli_overrides)

    return config


def parse_cli_overrides(args) -> Dict[str, Any]:
    """
    Parse argparse namespace to override dictionary.

    Only includes non-None values to allow YAML defaults to apply.

    Args:
        args: argparse.Namespace object

    Returns:
        Dictionary with nested structure for overrides
    """
    overrides = {'model': {}, 'training': {}}

    # Problem override
    if hasattr(args, 'problem') and args.problem is not None:
        overrides['problem'] = args.problem

    # Model overrides
    model_fields = ['num_feature', 'num_enhancement', 'feature_activation', 'enhancement_activation', 'bls_random_range']
    for field in model_fields:
        if hasattr(args, field) and getattr(args, field) is not None:
            overrides['model'][field] = getattr(args, field)

    # Training overrides
    training_fields = ['n_colloc', 'n_bc', 'n_init', 'n_ic_data', 'seed',
                       'cost_threshold', 'max_sub_iterations', 'visualize',
                       'points_verbose', 'save_model', 'results_dir']
    for field in training_fields:
        if hasattr(args, field) and getattr(args, field) is not None:
            overrides['training'][field] = getattr(args, field)

    # Clean up empty dicts
    if not overrides['model']:
        del overrides['model']
    if not overrides['training']:
        del overrides['training']

    return overrides


def save_resolved_config(
    problem_name: str,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    save_dir: str,
    filename: str = 'config.yaml'
) -> str:
    """
    Save the fully resolved configuration used for an experiment.

    This allows exact reproduction of experiments.

    Args:
        problem_name: Name of the problem being solved
        model_cfg: Model configuration used
        train_cfg: Training configuration used
        save_dir: Directory to save config
        filename: Output filename

    Returns:
        Path to saved config file
    """
    config = {
        'problem': problem_name,
        'model': model_config_to_dict(model_cfg),
        'training': training_config_to_dict(train_cfg),
    }

    save_path = Path(save_dir) / filename
    save_yaml(config, str(save_path))

    return str(save_path)
