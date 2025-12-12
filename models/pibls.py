"""
Physics-Informed Broad Learning System (PIBLS) model.

This module provides the PIBLS neural network architecture that combines
Broad Learning System (BLS) with physics-informed constraints for solving PDEs.

Key differences from standard BLS:
- No training labels; constraints come from PDE residuals
- Output weights solved via physics-constrained least squares
- Derivative calculations handled by the solver (separation of concerns)

Architecture:
    Input (x, t) -> Feature Layer -> Enhancement Layer -> [H matrix] -> Output weights -> u(x,t)

The H matrix (feature + enhancement outputs) is computed by this class.
The solver uses H to build residual equations and solve for output weights.
"""
import torch
import numpy as np

from models.bls import NodeGenerator
from configs.config import ModelConfig


class InputNormalizer:
    """Normalizes input tensors from [x_min, x_max] to [-1, 1]."""

    def __init__(self, x_min: torch.Tensor, x_max: torch.Tensor):
        self.x_min = x_min
        self.x_max = x_max
        self.x_range = self.x_max - self.x_min
        # Avoid division by zero for constant dimensions
        self.x_range[self.x_range < 1e-10] = 1.0

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map x from [x_min, x_max] to [-1, 1]."""
        return 2 * (x - self.x_min) / self.x_range - 1

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Map x from [-1, 1] back to [x_min, x_max]."""
        return (x_norm + 1) / 2 * self.x_range + self.x_min


class PIBLS(torch.nn.Module):
    """
    Physics-Informed Broad Learning System for solving PDEs.

    Architecture:
        Input (x, t) -> Feature Layer -> Enhancement Layer -> [H matrix] -> Output weights -> u(x,t)
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.cfg = model_cfg
        self.mapping_generator = NodeGenerator(random_range=self.cfg.bls_random_range)
        self.enhancement_generator = NodeGenerator(whiten=self.cfg.bls_enhance_whiten, random_range=self.cfg.bls_random_range)
        self.weights = None  # Solved during training by PIBLSSolver

    def setup_network(self, data: torch.Tensor):
        """Initialize random weights for both layers using sample input data."""
        data_np = data.detach().cpu().numpy()

        # Auto-configure feature hidden dimension if needed
        if self.cfg.feature_hidden_dim == 'auto':
            self.cfg.feature_hidden_dim = data_np.shape[1]

        # Generate feature layer weights and get transformed output
        mapping_data = self.mapping_generator.generator_nodes(
            data_np, self.cfg.num_feature, self.cfg.feature_hidden_dim, self.cfg.feature_activation
        )

        # Auto-configure enhancement hidden dimension if needed
        if self.cfg.enhancement_hidden_dim == 'auto':
            self.cfg.enhancement_hidden_dim = mapping_data.shape[1]

        # Generate enhancement layer weights
        self.enhancement_generator.generator_nodes(
            mapping_data, self.cfg.num_enhancement, self.cfg.enhancement_hidden_dim, self.cfg.enhancement_activation
        )

        print("PIBLS network initialized.")

    def get_output_layer_input(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute concatenated feature and enhancement output H(X).

        H(X) = [feature_output | enhancement_output]

        The solver uses H to build: H @ weights = targets (for BCs, ICs)
        and L(H @ weights) = f(x) (for PDE residual).
        """
        X_np = X.detach().cpu().numpy()

        # Transform through feature layer
        mapped = self.mapping_generator.transform(X_np)

        # Transform through enhancement layer
        enhanced = self.enhancement_generator.transform(mapped)

        # Concatenate feature and enhancement outputs
        result = torch.from_numpy(np.hstack([mapped, enhanced])).to(self.cfg.dtype)
        return result

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using the learned weights (must be called after training)."""
        H = self.get_output_layer_input(X)
        result = H @ self.weights
        return result

    def state_dict(self):
        """Extract model state for saving."""
        state = {
            'output_weights': self.weights.clone() if self.weights is not None else None,
            'mapping_generator': self.mapping_generator.state_dict(),
            'enhancement_generator': self.enhancement_generator.state_dict(),
            'config': {
                'num_feature': self.cfg.num_feature,
                'num_enhancement': self.cfg.num_enhancement,
                'feature_activation': self.cfg.feature_activation,
                'enhancement_activation': self.cfg.enhancement_activation,
                'feature_hidden_dim': self.cfg.feature_hidden_dim,
                'enhancement_hidden_dim': self.cfg.enhancement_hidden_dim,
                'dtype': str(self.cfg.dtype).replace('torch.', ''),
                'bls_random_range': self.cfg.bls_random_range,
                'bls_enhance_whiten': self.cfg.bls_enhance_whiten,
            }
        }
        return state

    def load_state_dict(self, state_dict):
        """Restore model state from saved state."""
        # Restore output weights
        if state_dict['output_weights'] is not None:
            self.weights = state_dict['output_weights'].clone().to(self.cfg.dtype)
        else:
            self.weights = None

        # Restore generator states
        self.mapping_generator.load_state_dict(state_dict['mapping_generator'])
        self.enhancement_generator.load_state_dict(state_dict['enhancement_generator'])
