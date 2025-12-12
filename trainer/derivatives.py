"""
Derivative computation for Physics-Informed Broad Learning Systems.

This module encapsulates all derivative calculations, providing a clean interface
between the solver and the model's internal structure. This separation ensures:
1. The solver doesn't need to know about model internals (weights, generators)
2. Derivative computation logic is centralized and testable
3. Changes to model architecture don't require solver modifications
"""
import torch
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple, Any

from physics.baseproblem import PDETerm, NonlinearTerm


class DerivativeCalculator:
    """
    Factory for retrieving pre-defined derivative functions of activation functions.

    The BLS network uses analytical derivatives for computing PDE residuals.
    This class provides derivatives up to 2nd order for supported activations.

    Supported activations: 'tanh', 'sin'

    Usage:
        deriv_func = DerivativeCalculator.get('tanh', order=1)
        result = deriv_func(Z)  # Z is pre-activation values
    """
    _registry = {
        "tanh": {
            0: torch.tanh,                                    # tanh(Z)
            1: lambda Z: 1 - torch.tanh(Z)**2,               # sech^2(Z)
            2: lambda Z: -2 * torch.tanh(Z) * (1 - torch.tanh(Z)**2),  # -2*tanh*sech^2
        },
        "sin": {
            0: torch.sin,           # sin(Z)
            1: torch.cos,           # cos(Z)
            2: lambda Z: -torch.sin(Z),  # -sin(Z)
        }
    }

    @classmethod
    def get(cls, activation: str, order: int) -> Callable:
        """
        Get the derivative function for given activation and order.

        Args:
            activation: Name of activation function ('tanh', 'sin')
            order: Derivative order (0=function, 1=first derivative, 2=second derivative)

        Returns:
            Callable that computes the derivative given pre-activation values
        """
        if activation not in cls._registry or order not in cls._registry[activation]:
            raise NotImplementedError(f"Derivative of order {order} for '{activation}' is not implemented.")
        return cls._registry[activation][order]


class DerivativeComputer:
    """
    Computes derivatives through the BLS network for physics-informed residuals.

    This class encapsulates all derivative calculations, hiding the model's
    internal structure from the solver. It computes:
    - Linear term derivatives (u_x, u_xx, u_t, etc.)
    - Nonlinear term residuals (sin(u), u*u_x, u^2)
    - Jacobians for nonlinear optimization

    Attributes:
        model: The PIBLS model (accessed through clean interfaces)
        model_cfg: Model configuration
        normalizer: Input normalizer for derivative scaling
    """

    def __init__(self, model, model_cfg, normalizer=None):
        """
        Initialize the derivative computer.

        Args:
            model: PIBLS model instance
            model_cfg: ModelConfig with architecture parameters
            normalizer: Optional InputNormalizer for coordinate scaling
        """
        self.model = model
        self.model_cfg = model_cfg
        self.normalizer = normalizer

        # Computation caches
        self._input_feature_cache: Dict[bytes, np.ndarray] = {}
        self._out_feature_cache: Dict[bytes, np.ndarray] = {}
        self._input_enhance_cache: Dict[bytes, np.ndarray] = {}
        self._terms_cache: Dict[Tuple, torch.Tensor] = {}

        # Precomputed basis matrices
        self._H_pred_nonlinear_colloc: Optional[torch.Tensor] = None
        self._H_dudx: Optional[torch.Tensor] = None

    def clear_cache(self):
        """Clear all computation caches."""
        self._input_feature_cache.clear()
        self._out_feature_cache.clear()
        self._input_enhance_cache.clear()
        self._terms_cache.clear()

    def precompute_nonlinear_bases(self, colloc_norm: torch.Tensor, nonlinear_terms: List[NonlinearTerm]):
        """
        Precompute basis matrices needed for nonlinear term evaluation.

        Args:
            colloc_norm: Normalized collocation points
            nonlinear_terms: List of nonlinear terms in the PDE
        """
        if colloc_norm.numel() == 0 or not nonlinear_terms:
            return

        # Basis for u prediction at collocation points
        self._H_pred_nonlinear_colloc = self.model.get_output_layer_input(colloc_norm)

        # Pre-compute basis for u*ux term if needed
        for term in nonlinear_terms:
            if term.term_type == 'u*ux':
                dudx_pde_term = PDETerm(var_index=term.var_index, derivative_order=1, coefficient=1.0, weight_power=1)
                self._H_dudx = self.calculate_linear_terms(colloc_norm, [dudx_pde_term])

    def _compute_feature_residual(self, term: PDETerm, input_feat: torch.Tensor,
                                  scaling_factor: float, coefficient: float) -> torch.Tensor:
        """
        Compute feature layer contribution to derivative.

        Args:
            term: PDE term specifying derivative order and variable
            input_feat: Pre-activation values for feature layer
            scaling_factor: Normalization scaling factor
            coefficient: Term coefficient

        Returns:
            Feature layer contribution tensor
        """
        W_var_feature = self.model.mapping_generator.get_feature_var_weights(term.var_index)
        feature_deriv = DerivativeCalculator.get(self.model_cfg.feature_activation, term.derivative_order)
        feature_contribution = feature_deriv(input_feat) * (W_var_feature ** term.weight_power) * coefficient * scaling_factor
        return feature_contribution

    def _compute_enhancement_residual(self, term: PDETerm, input_feat: torch.Tensor,
                                      input_enhance: torch.Tensor, scaling_factor: float,
                                      coefficient: float) -> torch.Tensor:
        """
        Compute enhancement layer contribution to derivative.

        Args:
            term: PDE term specifying derivative order and variable
            input_feat: Pre-activation values for feature layer
            input_enhance: Pre-activation values for enhancement layer
            scaling_factor: Normalization scaling factor
            coefficient: Term coefficient

        Returns:
            Enhancement layer contribution tensor
        """
        order = term.derivative_order
        if order == 0:
            enhance_deriv = DerivativeCalculator.get(self.model_cfg.enhancement_activation, 0)
            return enhance_deriv(input_enhance) * coefficient * scaling_factor

        elif order == 1:
            feature_1st_deriv = DerivativeCalculator.get(self.model_cfg.feature_activation, 1)
            enhance_1st_deriv = DerivativeCalculator.get(self.model_cfg.enhancement_activation, 1)
            W_var_feature = self.model.mapping_generator.get_feature_var_weights(term.var_index)
            W_enhance = self.model.enhancement_generator.get_enhance_weights()
            feature_1st_unscaled = feature_1st_deriv(input_feat) * W_var_feature
            enhance_contribution = enhance_1st_deriv(input_enhance) * (feature_1st_unscaled @ W_enhance)
            return enhance_contribution * coefficient * scaling_factor

        elif order == 2:
            feature_1st_deriv = DerivativeCalculator.get(self.model_cfg.feature_activation, 1)
            enhance_1st_deriv = DerivativeCalculator.get(self.model_cfg.enhancement_activation, 1)
            feature_2nd_deriv = DerivativeCalculator.get(self.model_cfg.feature_activation, 2)
            enhance_2nd_deriv = DerivativeCalculator.get(self.model_cfg.enhancement_activation, 2)

            W_var_feature = self.model.mapping_generator.get_feature_var_weights(term.var_index)
            W_enhance = self.model.enhancement_generator.get_enhance_weights()

            t1 = (feature_1st_deriv(input_feat) * W_var_feature) @ W_enhance
            t2 = feature_2nd_deriv(input_feat) * (W_var_feature ** term.weight_power)
            enhance_contribution = enhance_2nd_deriv(input_enhance) * (t1 ** 2) + \
                                   enhance_1st_deriv(input_enhance) * (t2 @ W_enhance)
            return enhance_contribution * coefficient * scaling_factor

        else:
            raise ValueError(f"Enhancement layer currently supports only 0-2 order derivatives, got: {order}")

    def calculate_linear_terms(self, X: torch.Tensor, terms: List[PDETerm]) -> torch.Tensor:
        """
        Compute the basis functions for derivative terms.

        This is the main method for computing H matrices for linear differential
        operators. Given input points X and a list of PDE terms, it returns the
        basis matrix H such that H @ weights gives the operator applied to the
        BLS output.

        Args:
            X: Input tensor of normalized coordinates, shape (n_points, input_dim)
            terms: List of PDETerm objects defining the differential operator

        Returns:
            Basis matrix H of shape (n_points, n_weights)
        """
        X_np = X.detach().cpu().numpy()

        terms_key = tuple(
            (term.derivative_order, term.var_index, term.weight_power,
             term.coefficient.__name__ if callable(term.coefficient) else term.coefficient)
            for term in terms
        )

        cache_key = (X.cpu().numpy().tobytes(), terms_key)

        # Check cache
        if cache_key in self._terms_cache:
            return self._terms_cache[cache_key]

        # Compute pre-activation values through feature layer
        if cache_key in self._input_feature_cache:
            input_feat = self._input_feature_cache[cache_key]
        else:
            input_feat = self.model.mapping_generator.compute_node_inputs(X_np)
            self._input_feature_cache[cache_key] = input_feat

        if cache_key in self._out_feature_cache:
            out_feat = self._out_feature_cache[cache_key]
        else:
            out_feat = self.model.mapping_generator.transform(X_np)
            self._out_feature_cache[cache_key] = out_feat

        # Compute pre-activation values through enhancement layer
        if cache_key in self._input_enhance_cache:
            input_enh = self._input_enhance_cache[cache_key]
        else:
            input_enh = self.model.enhancement_generator.compute_node_inputs(out_feat)
            self._input_enhance_cache[cache_key] = input_enh

        # Convert to tensors
        input_feat = torch.from_numpy(input_feat).to(self.model_cfg.dtype)
        input_enh = torch.from_numpy(input_enh).to(self.model_cfg.dtype)
        input_final = torch.hstack([input_feat, input_enh])
        H_terms = torch.zeros_like(input_final)

        # Denormalize X for callable coefficients
        X_denorm = self.normalizer.denormalize(X) if self.normalizer else X

        for term in terms:
            scaling_factor = (2.0 / self.normalizer.x_range[term.var_index]) ** term.derivative_order if self.normalizer else 1.0

            if callable(term.coefficient):
                coefficient = term.coefficient(X_denorm).view(-1, 1)
            else:
                coefficient = term.coefficient

            # Get feature residual
            feature_contribution = self._compute_feature_residual(term, input_feat, scaling_factor, coefficient)
            # Get enhancement residual
            enhance_contribution = self._compute_enhancement_residual(term, input_feat, input_enh,
                                                                      scaling_factor, coefficient)

            term_contribution = torch.hstack([feature_contribution, enhance_contribution])
            H_terms += term_contribution

        self._terms_cache[cache_key] = H_terms
        return H_terms

    def calculate_nonlinear_residual(self, u_pred: torch.Tensor, nl_terms: List[NonlinearTerm],
                                     weights: torch.Tensor) -> torch.Tensor:
        """
        Compute the value of nonlinear terms.

        Args:
            u_pred: Predicted u values at collocation points, shape (n_points, 1)
            nl_terms: List of NonlinearTerm objects
            weights: Current model weights for u*ux term

        Returns:
            Total nonlinear residual, shape (n_points, 1)
        """
        total_nl_residual = torch.zeros_like(u_pred)
        for nl_term in nl_terms:
            if nl_term.term_type == 'sin(u)':
                total_nl_residual += nl_term.coefficient * torch.sin(u_pred)
            elif nl_term.term_type == 'u*ux':
                if self._H_dudx is None:
                    raise ValueError("H_dudx matrix must be pre-computed for the 'u*ux' term.")
                du_dx = self._H_dudx @ weights
                total_nl_residual += u_pred * du_dx
            elif nl_term.term_type == 'u^2':
                total_nl_residual += nl_term.coefficient * (u_pred ** 2)
            else:
                raise NotImplementedError(f"Nonlinear term '{nl_term.term_type}' not implemented.")
        return total_nl_residual

    def calculate_nonlinear_jacobian(self, u_pred: torch.Tensor, nl_terms: List[NonlinearTerm],
                                     H_pred: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian of nonlinear terms with respect to weights.

        Args:
            u_pred: Predicted u values (n_points, 1)
            nl_terms: List of nonlinear terms
            H_pred: The H matrix for u prediction (n_points, n_weights)
            weights: Current model weights

        Returns:
            Jacobian matrix (n_points, n_weights)
        """
        n_points = u_pred.shape[0]
        n_weights = H_pred.shape[1]
        total_jacobian = torch.zeros((n_points, n_weights), dtype=u_pred.dtype, device=u_pred.device)

        for nl_term in nl_terms:
            if nl_term.term_type == 'sin(u)':
                # For sin(u): d/dw[sin(u)] = cos(u) * du/dw = cos(u) * H_pred
                cos_u = torch.cos(u_pred)
                jacobian = nl_term.coefficient * cos_u * H_pred
                total_jacobian += jacobian

            elif nl_term.term_type == 'u*ux':
                if self._H_dudx is None:
                    raise ValueError("H_dudx matrix must be pre-computed for the 'u*ux' term.")
                # For u*ux: d/dw[u*ux] = ux * du/dw + u * dux/dw
                du_dx = self._H_dudx @ weights
                jacobian = du_dx * H_pred + u_pred * self._H_dudx
                total_jacobian += jacobian

            elif nl_term.term_type == 'u^2':
                # For c*u^2: d/dw[c * u^2] = c * 2 * u * du/dw
                jacobian = nl_term.coefficient * 2 * u_pred * H_pred
                total_jacobian += jacobian

            else:
                raise NotImplementedError(f"Jacobian for nonlinear term '{nl_term.term_type}' not implemented.")

        return total_jacobian

    @property
    def H_pred_nonlinear_colloc(self) -> Optional[torch.Tensor]:
        """Get precomputed H matrix for nonlinear term evaluation."""
        return self._H_pred_nonlinear_colloc

    @property
    def H_dudx(self) -> Optional[torch.Tensor]:
        """Get precomputed H matrix for u_x derivative."""
        return self._H_dudx
