"""
Model storage utilities for saving and loading PIBLS models.

This module focuses on saving the essential model weights:
- Random initialization weights (feature/enhancement layers)
- Trained output weights
- Normalizer state (required for inference)

Configuration and metrics are already logged in experiment_results.csv,
so saved models are kept minimal and focused on weights for reproducibility.
"""
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from models.pibls import PIBLS
from configs.config import ModelConfig
from utils.config_loader import dict_to_model_config


MODEL_VERSION = "1.0"


def save_model(
    model: PIBLS,
    problem_name: str,
    normalizer: Optional[Any] = None,
    save_dir: str = "./save_models",
    filename: Optional[str] = None,
) -> str:
    """
    Save trained model with essential weights for reproducibility.

    Saves:
    - Output weights (trained)
    - Feature layer weights/biases (random initialization)
    - Enhancement layer weights/biases (random initialization)
    - Normalizer state (required for inference)
    - Config is included in model.state_dict()

    Args:
        model: Trained PIBLS model
        problem_name: Name of the problem solved
        normalizer: InputNormalizer instance (if used)
        save_dir: Directory to save model
        filename: Custom filename (default: {problem}_{timestamp}.pt)

    Returns:
        Path to saved model file
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = f"{problem_name}_{timestamp}.pt"

    save_path = save_dir / filename

    # Build model data - config is already in model.state_dict()['config']
    model_data = {
        'version': MODEL_VERSION,
        'problem_name': problem_name,
        'model_state': model.state_dict(),
    }

    # Add normalizer state (required for inference)
    if normalizer is not None:
        model_data['normalizer'] = {
            'x_min': normalizer.x_min.cpu().numpy().tolist(),
            'x_max': normalizer.x_max.cpu().numpy().tolist(),
            'x_range': normalizer.x_range.cpu().numpy().tolist(),
        }
    else:
        model_data['normalizer'] = None

    # Save model
    torch.save(model_data, save_path)
    print(f"Model saved to {save_path}")

    return str(save_path)


def load_model(
    model_path: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load model from file.

    Args:
        model_path: Path to .pt model file
        device: Device to load tensors to (default: CPU)

    Returns:
        Dictionary containing:
        - model: Reconstructed PIBLS model with loaded weights
        - model_config: ModelConfig instance
        - problem_name: Name of the problem
        - normalizer_state: Normalizer state dict (or None)
    """
    if device is None:
        device = 'cpu'

    model_data = torch.load(model_path, map_location=device, weights_only=False)

    # Verify version compatibility
    version = model_data.get('version', '0.0')
    if version != MODEL_VERSION:
        print(f"Warning: Model version {version} differs from current {MODEL_VERSION}")

    # Reconstruct model config - prefer model_state['config'], fallback to model_config for v1.0
    if 'config' in model_data.get('model_state', {}):
        model_cfg_dict = model_data['model_state']['config']
    elif 'model_config' in model_data:
        # Backwards compatibility with v1.0 format
        model_cfg_dict = model_data['model_config']
    else:
        raise ValueError(f"No config found in {model_path}")

    model_cfg = dict_to_model_config(model_cfg_dict)

    # Reconstruct model
    model = PIBLS(model_cfg)
    model.load_state_dict(model_data['model_state'])

    return {
        'model': model,
        'model_config': model_cfg,
        'problem_name': model_data['problem_name'],
        'normalizer_state': model_data.get('normalizer'),
    }


def load_model_for_inference(
    model_path: str,
    device: Optional[str] = None,
) -> Tuple[PIBLS, Optional[Dict[str, Any]]]:
    """
    Load model for inference only.

    This is a convenience function that returns just the model
    and normalizer state needed for making predictions.

    Args:
        model_path: Path to .pt model file
        device: Device to load tensors to

    Returns:
        Tuple of (model, normalizer_state)
        normalizer_state is None if no normalizer was used
    """
    result = load_model(model_path, device)
    return result['model'], result['normalizer_state']


def validate_model_config(
    model_path: str,
    model_cfg: ModelConfig,
) -> Dict[str, Any]:
    """
    Validate saved model config matches current config and return model data.

    This function loads a model and verifies that its model configuration
    is compatible with the provided model configuration. This ensures that
    a model trained with specific architecture settings is loaded correctly.

    Args:
        model_path: Path to .pt model file
        model_cfg: Current ModelConfig to validate against

    Returns:
        Dictionary containing model data with validated model

    Raises:
        ValueError: If saved model config doesn't match current model_cfg
    """
    result = load_model(model_path)
    saved_model_cfg = result['model_config']

    # Check critical config parameters match
    mismatches = []
    if saved_model_cfg.num_feature != model_cfg.num_feature:
        mismatches.append(f"num_feature: saved={saved_model_cfg.num_feature}, current={model_cfg.num_feature}")
    if saved_model_cfg.num_enhancement != model_cfg.num_enhancement:
        mismatches.append(f"num_enhancement: saved={saved_model_cfg.num_enhancement}, current={model_cfg.num_enhancement}")
    if saved_model_cfg.feature_activation != model_cfg.feature_activation:
        mismatches.append(f"feature_activation: saved={saved_model_cfg.feature_activation}, current={model_cfg.feature_activation}")
    if saved_model_cfg.enhancement_activation != model_cfg.enhancement_activation:
        mismatches.append(f"enhancement_activation: saved={saved_model_cfg.enhancement_activation}, current={model_cfg.enhancement_activation}")

    if mismatches:
        raise ValueError(f"Model config mismatch:\n  " + "\n  ".join(mismatches))

    print(f"Model config validated: {model_path}")
    return result
