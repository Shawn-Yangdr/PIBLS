"""
Visualization utilities for PIBLS solutions and point distributions.

This module provides:
- Visualizer: Plots predicted vs exact solutions and error distributions
- PointsVisualizer: Visualizes collocation, boundary, and initial points
"""
import os
import re
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from configs.config import TrainingConfig
from utils.common import save_plot_data


class Visualizer:
    """
    Handles plotting of solutions and errors for 1D and 2D problems.

    Supports:
    - 1D spatial problems: line plots with exact vs predicted
    - 2D spatial problems: 3D surface plots
    - Time-dependent problems: time slice plots with reference data
    """

    def __init__(self, problem, model, normalizer, train_cfg: TrainingConfig):
        """
        Initialize the visualizer.

        Args:
            problem: Problem instance with domain and exact solution info
            model: Trained PIBLS model for making predictions
            normalizer: InputNormalizer for coordinate transformation
            train_cfg: Training configuration with output settings
        """
        self.problem = problem
        self.model = model
        self.normalizer = normalizer
        self.train_cfg = train_cfg
        self.results_dir = train_cfg.results_dir
        # Get dtype from model config, not training config
        self.dtype = model.cfg.dtype
        os.makedirs(self.results_dir, exist_ok=True)

    def plot_solution(self):
        """Generic plot dispatcher."""
        print(f"Visualizing for {self.problem.input_dim}D problem: '{self.problem.name}'")
        if self.problem.input_dim == 1:
            self._plot_1d()
        elif self.problem.input_dim == 2:
            # Fisher problems use time slice visualization with reference data
            if 'Fisher' in self.problem.name:
                print("Generating time slice plot for Fisher problem.")
                self.plot_time_slices()
            else:
                print("Generating 2D heatmap plot.")
                self.plot_2d_heatmap()
        else:
            print(f"Visualization for {self.problem.input_dim}D not implemented.")

    def _plot_1d(self, n_points: int = 500):
        domain = self.problem.domain()
        x_test = torch.linspace(domain[0], domain[1], n_points, dtype=self.dtype).view(-1, 1)
        x_norm = self.normalizer.normalize(x_test)

        u_pred = self.model.predict(x_norm)
        u_exact = self.problem.exact_solution(x_test)
        error = u_exact - u_pred

        plt.figure(figsize=(7, 12))
        # Solution plot
        plt.subplot(2, 1, 1)

        plt.plot(x_test, u_exact, 'r-', label='Exact Solution', linewidth=2)
        plt.plot(x_test, u_pred, 'b--', label='PIBLS Prediction', linewidth=2)
        plt.title(f'{self.problem.name} - Solution')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True, color='#DDDDDD')

        # Error plot
        plt.subplot(2, 1, 2)
        plt.plot(x_test, error, color='#2C2C2C')
        plt.title('Error')
        plt.xlabel('x')
        plt.ylabel('Error')
        plt.grid(True, color='#DDDDDD')

        problem_name = self.problem.name.replace(" ", "_")
        params = f"feat{self.model.cfg.num_feature}_enh{self.model.cfg.num_enhancement}"
        params += f"_col{self.train_cfg.n_colloc}_delta{self.train_cfg.delta:.2f}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem_name}_{params}_{timestamp}.png"

        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=600)

        # Save plot data as npz file
        npz_dir = os.path.join(self.results_dir, "plot_Data")
        save_plot_data(npz_dir, problem_name,
                       x_test=x_test, u_exact=u_exact, u_pred=u_pred, error=error)
        plt.close()
        print(f"Saved visualization to {save_path}")

    def plot_time_slices(self, ref_data: Optional[Dict[str, Any]] = None):
        """
        Plot time-dependent solutions with reference data for Fisher problems.

        This method is specifically designed for Fisher-KPP problems that have
        external reference data for comparison. For other 2D problems, use
        plot_2d_heatmap() instead.

        Args:
            ref_data: Optional dictionary containing reference data with keys:
                - 'x': x coordinates (1D array)
                - 't': time points (1D array)
                - 'p': cell density values, shape (n_times, n_points)
                If None and problem has ic_data_path, will attempt to load from there.
        """
        # Get visualization config from problem if available
        viz_config = self._get_visualization_config()

        # Legacy support: load data if not provided
        if ref_data is None:
            from datasets.data_loader import ExternalDataLoader
            ic_path = getattr(self.problem, 'ic_data_path', None)
            if ic_path is None:
                print("Warning: No reference data provided and no ic_data_path found. Skipping visualization.")
                return
            ref_data = ExternalDataLoader.load_visualization_data(ic_path)

        x = ref_data['x'].squeeze()      # x coordinates
        t = ref_data['t'].squeeze()      # time points
        p = ref_data['p']                # cell density, shape (5, n)

        # Get visualization parameters
        t_snapshots = viz_config.get('t_snapshots', list(t))
        x_range = viz_config.get('x_range', (x.min(), x.max()))
        n_plot_points = viz_config.get('n_plot_points', 200)
        x_label = viz_config.get('x_label', 'x')
        y_label = viz_config.get('y_label', 'u')
        y_scale = viz_config.get('y_scale', 1.0)
        y_limits = viz_config.get('y_limits', None)
        x_limits = viz_config.get('x_limits', None)

        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
        markers = ['x', 'o', 's', 'D', '^']
        labels = [f'{int(ts)}h' for ts in t_snapshots]

        plt.figure(figsize=(14, 5))

        # Plot reference data points
        for i in range(len(t)):
            plt.plot(x, p[i], color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], linestyle='None', markersize=6)

        # Plot model predictions at each time snapshot
        all_predictions = []
        for i, t_val in enumerate(t_snapshots):
            x_test = torch.linspace(x_range[0], x_range[1], n_plot_points, dtype=self.dtype)
            t_test = torch.full_like(x_test, t_val)
            xt_test = torch.stack([x_test, t_test], dim=1)

            xt_norm = self.normalizer.normalize(xt_test)
            u_pred = self.model.predict(xt_norm)
            all_predictions.append(u_pred.cpu().numpy())
            plt.plot(x_test.cpu().numpy(), u_pred,
                    label=f"{labels[i]}", color=colors[i % len(colors)],
                    linestyle='-', linewidth=2)

        # Format y-axis with scientific notation if scale provided
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if y_scale != 1.0:
            plt.gca().yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda val, _: f"{val * y_scale:.1f}")
            )
            plt.gca().yaxis.offsetText.set_visible(False)

        if x_limits:
            plt.xlim(*x_limits)
        if y_limits:
            plt.ylim(*y_limits)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1),
            ncol=5,
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=14
        )
        plt.tight_layout()

        # Generate filename from problem name
        match = re.search(r'\d+', self.problem.name)
        num = match.group(0) if match else ''
        filename = f"Fisher{num}.png" if num else f"{self.problem.name.replace(' ', '_')}_time_slices.png"
        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=600)
        print(f"Saved visualization to {save_path}")

        # Save plot data for reproduction
        npz_filename = f"Fisher{num}_plot_data" if num else f"{self.problem.name}_plot_data"
        npz_dir = os.path.join(self.results_dir, "plot_Data")
        save_plot_data(npz_dir, npz_filename,
                       ref_x=x, ref_t=t, ref_p=p,
                       pred_x=x_test, pred_t_snapshots=np.array(t_snapshots),
                       pred_u_all=np.array(all_predictions))
        plt.close()

    def _get_visualization_config(self) -> Dict[str, Any]:
        """
        Get visualization configuration from problem or use defaults.

        Returns:
            Dictionary with visualization parameters
        """
        # Check if problem has custom visualization config
        if hasattr(self.problem, 'get_visualization_config'):
            return self.problem.get_visualization_config()

        # Default configuration
        x_domain, t_domain = self.problem.domain()
        return {
            't_snapshots': [t_domain[0], (t_domain[0] + t_domain[1]) / 2, t_domain[1]],
            'x_range': (x_domain[0], x_domain[1]),
            'x_label': 'x',
            'y_label': 'u(x, t)',
            'y_scale': 1.0,
            'y_limits': None,
            'x_limits': None,
            'n_plot_points': 200,
        }

    def _plot_2d_slices(self):
        """Plot time slices for time-dependent problems."""
        viz_config = self._get_visualization_config()
        x_domain, t_domain = self.problem.domain()

        t_snapshots = viz_config.get('t_snapshots',
                                      [t_domain[0], (t_domain[0] + t_domain[1]) / 2, t_domain[1]])

        for t in t_snapshots:
            x_test = torch.linspace(x_domain[0], x_domain[1], 200, dtype=self.dtype)
            t_test = torch.full_like(x_test, t)
            xt_test = torch.stack([x_test, t_test], dim=1)

            xt_norm = self.normalizer.normalize(xt_test)
            u_pred = self.model.predict(xt_norm)
            u_exact = self.problem.exact_solution(xt_test)
            error = u_exact - u_pred

            assert u_pred.shape == u_exact.shape, "Prediction and exact solution shapes do not match."

            plt.figure(figsize=(12, 5))

            # Solution plot
            plt.subplot(1, 2, 1)
            plt.plot(x_test, u_exact, 'r-', label='Exact', linewidth=2)
            plt.plot(x_test, u_pred, 'b--', label='PIBLS', linewidth=2)
            plt.title(f'{self.problem.name} at t={t:.2f}')
            plt.xlabel('x')
            plt.ylabel('u(x, t)')
            plt.legend()
            plt.grid(True)

            # Error plot
            plt.subplot(1, 2, 2)
            plt.plot(x_test, error, 'k-')
            plt.title(f'Error at t={t:.2f}')
            plt.xlabel('x')
            plt.ylabel('Error')
            plt.grid(True)

            filename = f"{self.problem.name.replace(' ', '_')}_t_{t:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = os.path.join(self.results_dir, filename)
            plt.savefig(save_path, dpi=600)
            plt.close()
            print(f"Saved visualization to {save_path}")

    def _plot_2d_surface(self, n_points: int = 200):
        """
        Generate a 2D surface/contour plot for spatial problems u(x, y).

        Creates three subplots:
        1. PIBLS prediction (3D surface)
        2. Exact solution (3D surface)
        3. Error distribution (2D heatmap)
        """
        # Create evaluation grid
        x_domain, y_domain = self.problem.domain()
        x_range = torch.linspace(x_domain[0], x_domain[1], n_points, dtype=self.dtype)
        y_range = torch.linspace(y_domain[0], y_domain[1], n_points, dtype=self.dtype)

        # Create 2D coordinate grid using meshgrid
        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')

        # Flatten grid to match model input format (N, 2)
        xy_test = torch.stack([X.flatten(), Y.flatten()], dim=1)
        xy_norm = self.normalizer.normalize(xy_test)

        # Get model predictions and exact solutions
        u_pred = self.model.predict(xy_norm)
        u_exact = self.problem.exact_solution(xy_test)
        U_pred_grid = u_pred.view(X.shape)
        U_exact_grid = u_exact.view(X.shape)
        Error_grid = U_pred_grid - U_exact_grid

        # Create figure with 3 subplots (prediction, exact, error)
        fig = plt.figure(figsize=(20, 6))

        # First subplot: Predicted solution (3D surface)
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_surface(X, Y, U_pred_grid, cmap='coolwarm', edgecolor='none')
        ax1.set_title('PIBLS Prediction')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u(x, y)')

        # Second subplot: Exact solution (3D surface)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_surface(X, Y, U_exact_grid, cmap='coolwarm', edgecolor='none')
        ax2.set_title('Exact Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u(x, y)')

        # Third subplot: Error (2D heatmap)
        ax3 = fig.add_subplot(1, 3, 3)
        c = ax3.pcolormesh(X, Y, Error_grid, cmap='coolwarm', shading='gouraud')
        fig.colorbar(c, ax=ax3)  # Add colorbar for heatmap
        ax3.set_title('Error')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')

        fig.tight_layout()

        # Save figure
        problem_name = self.problem.name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{problem_name}_{timestamp}.png"
        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=600)

        # Save plot data as npz file
        npz_dir = os.path.join(self.results_dir, "plot_Data")
        save_plot_data(npz_dir, problem_name,
                       x_range=x_range, y_range=y_range,
                       U_pred_grid=U_pred_grid, U_exact_grid=U_exact_grid,
                       Error_grid=Error_grid)

        plt.close()
        print(f"Saved 2D surface plot to {save_path}")

    def plot_2d_heatmap(self, n_points: int = 200):
        """
        Generate a 2D heatmap visualization for spatial problems u(x, y).

        Creates three side-by-side heatmaps:
        1. PIBLS prediction
        2. Exact solution
        3. Error distribution

        This provides a cleaner 2D view compared to _plot_2d_surface which uses 3D plots.

        Args:
            n_points: Number of points along each axis for the evaluation grid
        """
        # Create evaluation grid
        x_domain, y_domain = self.problem.domain()
        x_range = torch.linspace(x_domain[0], x_domain[1], n_points, dtype=self.dtype)
        y_range = torch.linspace(y_domain[0], y_domain[1], n_points, dtype=self.dtype)

        # Create 2D coordinate grid
        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')

        # Flatten grid to match model input format (N, 2)
        xy_test = torch.stack([X.flatten(), Y.flatten()], dim=1)
        xy_norm = self.normalizer.normalize(xy_test)

        # Get model predictions and exact solutions
        u_pred = self.model.predict(xy_norm)
        u_exact = self.problem.exact_solution(xy_test)

        # Reshape to grid
        U_pred = u_pred.view(X.shape).detach().cpu().numpy()
        U_exact = u_exact.view(X.shape).detach().cpu().numpy()
        Error = (U_pred - U_exact)
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()

        # Get problem name for titles
        name = self.problem.name

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Prediction heatmap
        ax1 = axes[0]
        pcm1 = ax1.pcolormesh(X_np, Y_np, U_pred, cmap='coolwarm', shading='gouraud')
        cbar1 = fig.colorbar(pcm1, ax=ax1, pad=0.02)
        cbar1.ax.tick_params(direction='in', length=2)
        ax1.tick_params(axis='both', direction='in', length=2)
        ax1.set_title(f"{name} Prediction")
        ax1.set_xlabel("x", labelpad=0)
        ax1.set_ylabel("y", labelpad=0)

        # Exact solution heatmap
        ax2 = axes[1]
        pcm2 = ax2.pcolormesh(X_np, Y_np, U_exact, cmap='coolwarm', shading='gouraud')
        cbar2 = fig.colorbar(pcm2, ax=ax2, pad=0.02)
        cbar2.ax.tick_params(direction='in', length=2)
        ax2.tick_params(axis='both', direction='in', length=2)
        ax2.set_title(f"{name} Exact")
        ax2.set_xlabel("x", labelpad=0)
        ax2.set_ylabel("y", labelpad=0)

        # Error heatmap
        ax3 = axes[2]
        pcm3 = ax3.pcolormesh(X_np, Y_np, Error, cmap='coolwarm', shading='gouraud')
        cbar3 = fig.colorbar(pcm3, ax=ax3, pad=0.02)
        cbar3.ax.tick_params(direction='in', length=2)
        ax3.tick_params(axis='both', direction='in', length=2)
        ax3.set_title(f"{name} Error")
        ax3.set_xlabel("x", labelpad=0)
        ax3.set_ylabel("y", labelpad=0)

        fig.tight_layout()

        # Save figure
        problem_name = self.problem.name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem_name}_heatmap_{timestamp}.png"
        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=600)

        # Save plot data
        npz_dir = os.path.join(self.results_dir, "plot_Data")
        save_plot_data(npz_dir, f"{problem_name}_heatmap",
                       X=X_np, Y=Y_np, U_pred=U_pred, U_exact=U_exact, Error=Error)

        plt.close()
        print(f"Saved 2D heatmap to {save_path}")
        return save_path


class PointsVisualizer:
    """Mesh point visualization tool for debugging and analysis."""

    @staticmethod
    def visualize(colloc_points, bc_points, init_points, problem, save_dir="./results"):
        """
        Visualize the distribution of different point types.

        Args:
            colloc_points: Collocation point coordinates, shape (n_colloc, dim)
            bc_points: Boundary point coordinates, shape (n_bc, dim)
            init_points: Initial point coordinates, shape (n_init, dim)
            problem: Problem instance for name, dimension, and boundary type info
            save_dir: Directory to save the visualization

        Returns:
            Path to the saved image file
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem.name.replace(' ', '_')}_points_{timestamp}.png"
        save_path = os.path.join(save_dir, filename)

        dim = problem.input_dim if hasattr(problem, 'input_dim') else colloc_points.shape[1]

        if dim == 1:
            PointsVisualizer._visualize_1d(colloc_points, bc_points, init_points, problem, save_path)
        elif dim == 2:
            PointsVisualizer._visualize_2d(colloc_points, bc_points, init_points, problem, save_path)
        else:
            print(f"Point distribution visualization only supports 1D and 2D problems, current is {dim}D")
            return None

        print(f"Point distribution visualization saved to {save_path}")
        return save_path

    @staticmethod
    def _visualize_1d(colloc_points, bc_points, init_points, problem, save_path):
        """Visualize point distribution for 1D problems."""
        plt.figure(figsize=(12, 6))

        # Plot collocation points
        if colloc_points.shape[0] > 0:
            plt.scatter(colloc_points[:, 0].cpu(), torch.zeros_like(colloc_points[:, 0]).cpu(),
                       color='blue', label='Collocation Points', alpha=0.5, s=30)

        # Plot boundary points
        if bc_points is not None and bc_points.shape[0] > 0:
            plt.scatter(bc_points[:, 0].cpu(), torch.zeros_like(bc_points[:, 0]).cpu(),
                       color='red', label='Boundary Points', alpha=0.8, s=50, marker='x')

        # Plot initial points
        if init_points is not None and init_points.shape[0] > 0:
            plt.scatter(init_points[:, 0].cpu(), torch.zeros_like(init_points[:, 0]).cpu(),
                       color='green', label='Initial Points', alpha=0.8, s=50, marker='^')

        plt.title(f'Point Distribution - {problem.name} (1D)')
        plt.xlabel('x')
        plt.yticks([])  # Hide y-axis ticks
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()

    @staticmethod
    def _visualize_2d(colloc_points, bc_points, init_points, problem, save_path):
        """Visualize point distribution for 2D problems."""
        plt.figure(figsize=(12, 10))

        # Plot collocation points
        if colloc_points.shape[0] > 0:
            plt.scatter(colloc_points[:, 0].cpu(), colloc_points[:, 1].cpu(),
                       color='blue', label='Collocation Points', alpha=0.3, s=20)

        # Plot boundary points
        if bc_points is not None and bc_points.shape[0] > 0:
            # Highlight point pairs for periodic boundaries
            boundary_type = getattr(problem, 'boundary_type', 'dirichlet')
            problem_type = getattr(problem, 'problem_type', 'spatial')

            if boundary_type == 'periodic':
                n_half = bc_points.shape[0] // 2
                plt.scatter(bc_points[:n_half, 0].cpu(), bc_points[:n_half, 1].cpu(),
                           color='red', label='Left Boundary', alpha=0.8, s=40, marker='+')
                plt.scatter(bc_points[n_half:, 0].cpu(), bc_points[n_half:, 1].cpu(),
                           color='magenta', label='Right Boundary', alpha=0.8, s=40, marker='x')

                # Connect periodic boundary point pairs with dashed lines
                if problem_type == 'time-dependent':
                    max_lines = min(20, n_half)
                    step = max(1, n_half // max_lines)
                    for i in range(0, n_half, step):
                        plt.plot([bc_points[i, 0].cpu(), bc_points[i+n_half, 0].cpu()],
                                [bc_points[i, 1].cpu(), bc_points[i+n_half, 1].cpu()],
                                'k--', alpha=0.3, linewidth=0.5)
            else:
                plt.scatter(bc_points[:, 0].cpu(), bc_points[:, 1].cpu(),
                           color='red', label='Boundary Points', alpha=0.8, s=30, marker='x')

        # Plot initial points
        if init_points is not None and init_points.shape[0] > 0:
            plt.scatter(init_points[:, 0].cpu(), init_points[:, 1].cpu(),
                       color='green', label='Initial Points', alpha=0.8, s=40, marker='^')

        # Set title and axis labels based on problem type
        problem_type = getattr(problem, 'problem_type', 'spatial')
        if problem_type == 'time-dependent':
            plt.title(f'Point Distribution - {problem.name} (Time-Space Problem)')
            plt.xlabel('x')
            plt.ylabel('t')
        else:
            plt.title(f'Point Distribution - {problem.name} (2D Spatial Problem)')
            plt.xlabel('x')
            plt.ylabel('y')

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()
