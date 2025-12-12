"""
Physics-Informed Broad Learning System Solver.

Orchestrates point generation, feature extraction, residual assembly,
and linear/nonlinear optimization to find output weights.
"""
import torch
import numpy as np
import time
import os
import scipy.linalg

from models.pibls import PIBLS, InputNormalizer
from trainer.derivatives import DerivativeComputer
from physics.baseproblem import BaseProblem
from configs.config import ModelConfig, TrainingConfig
from datasets.data_loader import MeshGenerator
from utils.common import set_random_seeds, calculate_errors, log_results, nlsq_perturb
from utils.visualizer import Visualizer, PointsVisualizer
from utils.model_storage import save_model, validate_model_config


class PIBLSSolver:
    """Physics-Informed Broad Learning System Solver."""

    def __init__(self, problem: BaseProblem, model: PIBLS, model_cfg: ModelConfig,
                 train_cfg: TrainingConfig, model_path: str = None):
        self.problem = problem
        self.model = model
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model_path = model_path
        self.normalizer = None
        self._train_time = 0.0
        self._skip_training = False

        # If model path provided, validate config compatibility
        if model_path:
            self._validate_and_load_model(model_path)

        set_random_seeds(self.train_cfg.seed)

        # DerivativeComputer will be initialized after normalizer is set up
        self.derivative_computer = None

        # Precomputed basis matrices for different residual types
        self._H_pred_bc = None             # Basis for boundary conditions
        self._H_pred_init = None           # Basis for initial conditions

        self._setup_for_training()

    def _setup_for_training(self):
        """Setup mesh, normalizer, and model network for training."""
        print(f"problem is {self.problem.name}, input_dim is {self.problem.input_dim}")

        # Get all points from MeshGenerator
        self.colloc, self.bc, self.init, self.ic_data_points, self.ic_data_target = MeshGenerator.get_points(
            self.problem, self.train_cfg, dtype=self.model_cfg.dtype,
            boundary_filters=self.problem.get_boundary_filters()
        )

        if self.train_cfg.points_verbose:
            PointsVisualizer.visualize(self.colloc, self.bc, self.init, self.problem,
                                       save_dir=self.train_cfg.results_dir)

        print("point generate done")

        # Setup normalizer
        domain_min = torch.tensor([d[0] for d in (self.problem.domain() if self.problem.input_dim > 1 else [self.problem.domain()])], dtype=self.model_cfg.dtype)
        domain_max = torch.tensor([d[1] for d in (self.problem.domain() if self.problem.input_dim > 1 else [self.problem.domain()])], dtype=self.model_cfg.dtype)
        self.normalizer = InputNormalizer(domain_min, domain_max) if self.train_cfg.normalize_input else None

        # Initialize derivative computer with model and normalizer
        self.derivative_computer = DerivativeComputer(self.model, self.model_cfg, self.normalizer)

        # Normalize all points
        if self.normalizer:
            self.colloc_norm = self.normalizer.normalize(self.colloc)
            self.bc_norm = self.normalizer.normalize(self.bc)
            self.init_norm = self.normalizer.normalize(self.init)
            self.ic_data_points_norm = self.normalizer.normalize(self.ic_data_points) if self.ic_data_points.numel() > 0 else self.ic_data_points
        else:
            self.colloc_norm, self.bc_norm, self.init_norm = self.colloc, self.bc, self.init
            self.ic_data_points_norm = self.ic_data_points

        print("number of collocation points:", self.colloc.shape[0])
        print("number of boundary points:", self.bc.shape[0] if self.bc is not None else 0)
        print("number of grid-based initial points:", self.init.shape[0] if self.init is not None else 0)
        print("number of data-driven initial points:", self.ic_data_points.shape[0])

        # Setup model network
        all_points = torch.cat([self.colloc_norm, self.bc_norm, self.init_norm, self.ic_data_points_norm], dim=0)
        self.all_points = all_points
        self.model.setup_network(all_points)

        self.precomputed_residuals = {}
        self.visualizer = Visualizer(self.problem, self.model, self.normalizer, self.train_cfg)

    def _validate_and_load_model(self, model_path: str):
        """Validate model config matches current config and load weights."""
        result = validate_model_config(model_path, self.model_cfg)
        self._model_state = result['model'].state_dict()
        self._skip_training = True

    def _precompute_all_residuals(self):
        """Pre-compute basis matrix H and target vector K for all residual terms."""
        residual_definitions = self.problem.get_residual_terms(self.colloc_norm, self.bc_norm, self.init_norm)

        for term in residual_definitions:
            H_matrix = None
            points_norm = term.points

            # 1. Compute basis matrix H
            if term.operator_terms:  # If operator defined (e.g., u_xx, u_t)
                H_matrix = self.derivative_computer.calculate_linear_terms(points_norm, term.operator_terms)
            elif term.is_periodic:
                # Periodic BC: H = H_left - H_right (constraint: u_left = u_right)
                H_base = self.model.get_output_layer_input(points_norm)
                n_half = H_base.shape[0] // 2
                H_matrix = H_base[:n_half] - H_base[n_half:]
            else:
                # No operator, just evaluate u directly
                H_matrix = self.model.get_output_layer_input(points_norm)

            # 2. Compute target vector K
            if term.is_periodic:
                # Periodic BC target is always 0
                n_half = points_norm.shape[0] // 2
                K_target = torch.zeros((n_half, 1), dtype=self.model_cfg.dtype)
            else:
                points_denorm = self.normalizer.denormalize(points_norm) if self.normalizer else points_norm
                K_target = term.target_func(points_denorm).view(-1, 1)

            self.precomputed_residuals[term.name] = {'H': H_matrix, 'K': K_target}

        if self.ic_data_points_norm.numel() > 0:
            print(f"Precomputing residual for {self.ic_data_points_norm.shape[0]} data-driven IC points.")
            H_ic_data = self.model.get_output_layer_input(self.ic_data_points_norm)
            K_ic_data = self.ic_data_target  # This is the loaded 'u' data
            self.precomputed_residuals['ic_data'] = {'H': H_ic_data, 'K': K_ic_data}

        # 3. Precompute basis matrices for nonlinear terms
        if self.colloc_norm.numel() > 0 and self.problem.nonlinear_terms():
            self.derivative_computer.precompute_nonlinear_bases(
                self.colloc_norm, self.problem.nonlinear_terms()
            )

    def _residual_function(self, weights_np: np.ndarray):
        """Assemble total residual vector based on precomputed matrices."""
        self.model.weights = torch.from_numpy(weights_np).view(-1, 1).to(self.model_cfg.dtype)

        all_residuals = []

        # Process all residual terms in a unified loop
        for name, data in self.precomputed_residuals.items():
            # Compute linear part residual for all terms
            residual = data['H'] @ self.model.weights - data['K']

            # If PDE term with nonlinear components, add nonlinear residual
            if name == 'pde' and self.problem.nonlinear_terms():
                H_pred = self.derivative_computer.H_pred_nonlinear_colloc
                u_pred_colloc = H_pred @ self.model.weights
                nl_residual = self.derivative_computer.calculate_nonlinear_residual(
                    u_pred_colloc, self.problem.nonlinear_terms(), self.model.weights
                )
                residual += nl_residual

            all_residuals.append(residual)

        total_residual = torch.cat(all_residuals, dim=0).cpu().numpy().flatten()
        return total_residual

    def _jacobian_function(self, weights_np: np.ndarray):
        """Compute analytical Jacobian of the residual function."""
        self.model.weights = torch.from_numpy(weights_np).view(-1, 1).to(self.model_cfg.dtype)

        all_jacobians = []

        # Process each residual term's Jacobian
        for name, data in self.precomputed_residuals.items():
            # Linear part: Jacobian is just the H matrix
            jacobian = data['H']

            # If this is PDE term with nonlinear components, add nonlinear Jacobian
            if name == 'pde' and self.problem.nonlinear_terms():
                # Compute u_pred for nonlinear Jacobian calculation
                H_pred = self.derivative_computer.H_pred_nonlinear_colloc
                u_pred_colloc = H_pred @ self.model.weights

                # Calculate Jacobian of nonlinear terms
                nl_jacobian = self.derivative_computer.calculate_nonlinear_jacobian(
                    u_pred_colloc,
                    self.problem.nonlinear_terms(),
                    H_pred,
                    self.model.weights
                )
                jacobian = jacobian + nl_jacobian

            all_jacobians.append(jacobian)

        # Stack all Jacobians vertically
        total_jacobian = torch.cat(all_jacobians, dim=0).cpu().numpy()
        return total_jacobian

    def train(self):
        """Train the model or load weights from saved model if provided."""
        if self._skip_training:
            # Load weights from validated model
            self.model.load_state_dict(self._model_state)
            print(f"Loaded weights from saved model, skipping training.")
            return

        print(f"--- Starting Training for {self.problem.name} ---")
        start_time = time.time()

        self._precompute_all_residuals()

        any_H = next(iter(self.precomputed_residuals.values()))['H']
        num_weights = any_H.shape[1]

        if self.problem.nonlinear_terms():
            print("Computing linear solution as initial guess for nonlinear solver...")
            H_list = [res['H'] for res in self.precomputed_residuals.values()]
            K_list = [res['K'] for res in self.precomputed_residuals.values()]
            H = torch.cat(H_list, dim=0)
            K = torch.cat(K_list, dim=0)
            w0_linear, _, _, _ = scipy.linalg.lstsq(H.cpu().numpy(), K.cpu().numpy())
            w0 = w0_linear.flatten()
            print(f"Linear initial guess computed. Max weight: {np.max(np.abs(w0)):.4f}")
        else:
            w0 = np.zeros(num_weights)

        if not self.problem.nonlinear_terms():
            print("No nonlinear terms found, using linear least squares.")
            H_list = [res['H'] for res in self.precomputed_residuals.values()]
            K_list = [res['K'] for res in self.precomputed_residuals.values()]
            H = torch.cat(H_list, dim=0)
            K = torch.cat(K_list, dim=0)
            weights_np, _, _, _ = scipy.linalg.lstsq(H.cpu().numpy(), K.cpu().numpy())
            self.model.weights = torch.from_numpy(weights_np).view(-1, 1).to(self.model_cfg.dtype)
        else:
            print("Nonlinear terms found, using perturbed least squares.")
            result = nlsq_perturb(
                func=self._residual_function,
                jac=self._jacobian_function,
                x0=w0,
                delta=self.train_cfg.delta,
                cost_threshold=self.train_cfg.cost_threshold,
                max_sub_iterations=self.train_cfg.max_sub_iterations,
                verbose=1
            )
            self.model.weights = torch.from_numpy(result.x).view(-1, 1).to(self.model_cfg.dtype)
            print(f"w_max = {torch.max(self.model.weights)}, w_min = {torch.min(self.model.weights)}")

        self._train_time = time.time() - start_time
        print(f"--- Training Finished in {self._train_time:.2f} seconds ---")

        # Save after training (not when loading from saved model)
        self.save()

    def evaluate(self):
        """Evaluate model and print results."""
        metrics = self._compute_errors()
        print(f"\nEvaluation Results:")
        print(f"  Max Error:  {metrics[0]:.6e}")
        print(f"  Mean Error: {metrics[1]:.6e}")
        print(f"  L2 Error:   {metrics[2]:.6e}")
        return metrics

    def save(self):
        """Save model and log results."""
        metrics = self._compute_errors()
        elapsed_time = getattr(self, '_train_time', 0.0)

        log_path = os.path.join(self.train_cfg.results_dir, self.train_cfg.results_name)
        log_results(log_path, self.model_cfg, self.train_cfg, self.problem.name, metrics, elapsed_time)

        if self.train_cfg.save_model:
            models_dir = os.path.join(self.train_cfg.results_dir, "save_models")
            save_model(
                model=self.model,
                problem_name=self.problem.name,
                normalizer=self.normalizer,
                save_dir=models_dir,
            )

    def visualize(self):
        """Plot solution visualization."""
        self.visualizer.plot_solution()

    def _compute_errors(self):
        """Calculates max and L2 error against the exact solution."""
        # Use evaluation config from TrainingConfig
        if self.train_cfg.n_eval > 0:
            # Direct evaluation point count specified
            n_eval_colloc = self.train_cfg.n_eval
            n_eval_bc = 0
            n_eval_init = 0
        else:
            # Auto-compute based on training points
            multiplier = self.train_cfg.eval_multiplier
            max_points = self.train_cfg.eval_max_points

            def compute_eval_points(n_train):
                if n_train == 0:
                    return 0
                return max(1000, min(n_train * multiplier, max_points))

            n_eval_colloc = compute_eval_points(self.train_cfg.n_colloc)
            n_eval_bc = compute_eval_points(self.train_cfg.n_bc)
            n_eval_init = compute_eval_points(self.train_cfg.n_init)

        # Create evaluation config
        eval_cfg = TrainingConfig(
            n_colloc=n_eval_colloc,
            n_bc=n_eval_bc,
            n_init=n_eval_init,
            dtype=self.model_cfg.dtype
        )
        x_eval, _, _, _, _ = MeshGenerator.get_points(self.problem, eval_cfg)

        x_eval_norm = self.normalizer.normalize(x_eval) if self.normalizer else x_eval

        u_pred = self.model.predict(x_eval_norm)
        u_exact = self.problem.exact_solution(x_eval)

        assert u_pred.shape == u_exact.shape, f"Shape mismatch: u_pred.shape={u_pred.shape}, u_exact.shape={u_exact.shape}"

        return calculate_errors(u_pred, u_exact)[:3]
