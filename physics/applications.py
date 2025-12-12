"""
Application problems for PIBLS - real-world PDE problems.

These problems are typically data-driven (no exact solution) and include:
- Fisher-KPP equation for cell migration/population dynamics
"""
from abc import ABC, abstractmethod
import torch
from typing import List, Tuple, Union, Optional

from physics.baseproblem import BaseProblem, PDETerm, NonlinearTerm, ResidualTerm


# Default data paths (relative to project root)
DEFAULT_DATA_ROOT = "./data"


class FisherDATABase(BaseProblem):
    """
    Base class for data-driven Fisher-KPP problems with configurable parameters.

    PDE: dp/dt = gamma * p_xx + lambda1 * p - lambda2 * p^2
        => p_t - gamma * p_xx - lambda1 * p + lambda2 * p^2 = 0
    IC: u(x, 0) = from data file
    BC: u_x(0, t) = 0, u_x(L, t) = 0 (Neumann)

    Args:
        ic_data_path: Path to .mat file with initial condition data.
        gamma: Diffusion coefficient
        lambda1: Linear growth coefficient
        lambda2: Nonlinear coefficient
    """
    def __init__(self,
                 ic_data_path: Optional[str] = None,
                 gamma: float = -310.0,
                 lambda1: float = -0.044,
                 lambda2: float = 44/1.7):
        super().__init__()
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L = 1950.0
        self.T = 48.0
        self.A = 0.95
        self.w = 300.0
        self._ic_data_path = ic_data_path

    @property
    def name(self) -> str:
        return "Fisher-KPP Problem"

    @property
    def input_dim(self) -> int:
        return 2  # x, t

    @property
    def problem_type(self) -> str:
        return 'time-dependent'

    @property
    def ic_data_path(self) -> Optional[str]:
        """Path to the .mat file containing initial condition data."""
        return self._ic_data_path

    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # Domain is (x_domain, t_domain)
        return ((0.0, self.L), (0.0, self.T))

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """No exact solution is provided. Return zeros."""
        return torch.zeros_like(x[:, 0:1])

    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        """The PDE is homogeneous (source term is 0)."""
        return torch.zeros_like(x[:, 0:1])

    def linear_terms(self) -> List[PDETerm]:
        # PDE: p_t - gamma * p_xx - lambda1 * p - ... = 0
        return [
            # p_t (var_index=1 is t)
            PDETerm(derivative_order=1, var_index=1, weight_power=1, coefficient=1.0),
            # -gamma * p_xx (var_index=0 is x)
            PDETerm(derivative_order=2, var_index=0, weight_power=2, coefficient=self.gamma),
            # -lambda1 * p
            PDETerm(derivative_order=0, var_index=0, weight_power=0, coefficient=self.lambda1),
        ]

    def nonlinear_terms(self) -> List[NonlinearTerm]:
        # PDE: ... - lambda2 * p^2 = 0
        return [
            # -lambda2 * p^2
            NonlinearTerm(term_type='u^2', var_index=0, coefficient=self.lambda2)
        ]

    def initial_conditions(self, x_init: torch.Tensor) -> torch.Tensor:
        """ u(x, 0) = 1 - A * exp( -(x - L/2)^2 / w^2 ) """
        # This is the fallback if n_init > 0 and n_ic_data = 0
        x = x_init[:, 0:1] # x values (t is 0)
        L_half = self.L / 2.0
        w_sq = self.w**2
        exponent = -((x - L_half)**2) / w_sq
        return 1.0 - self.A * torch.exp(exponent)

    def neumann_boundary_target(self, x_bc: torch.Tensor) -> torch.Tensor:
        """Target for Neumann BCs is 0."""
        return torch.zeros_like(x_bc[:, 0:1])

    def get_residual_terms(self, colloc: torch.Tensor, bc: torch.Tensor, init: torch.Tensor) -> List[ResidualTerm]:
        """Override to implement Neumann BCs."""
        terms = []

        # 1. PDE Residual
        if colloc.numel() > 0:
            terms.append(ResidualTerm(
                name='pde',
                points=colloc,
                target_func=self.pde_source_term,
                operator_terms=self.linear_terms()
            ))

        # 2. Boundary Condition Residual
        if bc.numel() > 0:
            du_dx_operator = [PDETerm(derivative_order=1, var_index=0, weight_power=1)]
            terms.append(ResidualTerm(
                name='bc',
                points=bc,
                target_func=self.neumann_boundary_target,
                operator_terms=du_dx_operator
            ))

        # 3. Initial Condition Residual (grid-based fallback)
        if init.numel() > 0:
            terms.append(ResidualTerm(
                name='ic_grid',
                points=init,
                target_func=self.initial_conditions,
                operator_terms=None
            ))

        return terms

    def get_visualization_config(self) -> dict:
        """
        Return problem-specific visualization configuration.

        Override this method for visualization settings.
        """
        return {
            't_snapshots': [0, 12, 24, 36, 48],
            'x_range': (25, 1875),
            'x_label': "Position (μm)",
            'y_label': "Cell density (cells/μm²)",
            'y_scale': 1e3,  # Scale factor for y-axis display
            'y_limits': (-0.0001, 0.0021),
            'x_limits': (0, 1950),
            'n_plot_points': 200,
        }


class FisherDATA1000(FisherDATABase):
    """Fisher-KPP with data1000.mat initial conditions."""

    def __init__(self, ic_data_path: Optional[str] = None, data_root: str = DEFAULT_DATA_ROOT):
        # Use provided path or construct from data_root
        if ic_data_path is None:
            ic_data_path = f"{data_root}/data1000.mat"
        super().__init__(
            ic_data_path=ic_data_path,
            gamma=-310.0,
            lambda1=-0.044,
            lambda2=44/1.7
        )

    @property
    def name(self) -> str:
        return "Fisher-KPP Problem(1000)"


class FisherDATA1200(FisherDATABase):
    """Fisher-KPP with data1200.mat initial conditions."""

    def __init__(self, ic_data_path: Optional[str] = None, data_root: str = DEFAULT_DATA_ROOT):
        if ic_data_path is None:
            ic_data_path = f"{data_root}/data1200.mat"
        super().__init__(
            ic_data_path=ic_data_path,
            gamma=-250.0,
            lambda1=-0.044,
            lambda2=44/1.7
        )

    @property
    def name(self) -> str:
        return "Fisher-KPP Problem(1200)"


class FisherDATA1400(FisherDATABase):
    """Fisher-KPP with data1400.mat initial conditions."""

    def __init__(self, ic_data_path: Optional[str] = None, data_root: str = DEFAULT_DATA_ROOT):
        if ic_data_path is None:
            ic_data_path = f"{data_root}/data1400.mat"
        super().__init__(
            ic_data_path=ic_data_path,
            gamma=-720.0,
            lambda1=-0.048,
            lambda2=48/1.7
        )

    @property
    def name(self) -> str:
        return "Fisher-KPP Problem(1400)"


class FisherDATA1600(FisherDATABase):
    """Fisher-KPP with data1600.mat initial conditions."""

    def __init__(self, ic_data_path: Optional[str] = None, data_root: str = DEFAULT_DATA_ROOT):
        if ic_data_path is None:
            ic_data_path = f"{data_root}/data1600.mat"
        super().__init__(
            ic_data_path=ic_data_path,
            gamma=-570.0,
            lambda1=-0.049,
            lambda2=49/1.7
        )

    @property
    def name(self) -> str:
        return "Fisher-KPP Problem(1600)"


class FisherDATA1800(FisherDATABase):
    """Fisher-KPP with data1800.mat initial conditions."""

    def __init__(self, ic_data_path: Optional[str] = None, data_root: str = DEFAULT_DATA_ROOT):
        if ic_data_path is None:
            ic_data_path = f"{data_root}/data1800.mat"
        super().__init__(
            ic_data_path=ic_data_path,
            gamma=-760.0,
            lambda1=-0.054,
            lambda2=54/1.7
        )

    @property
    def name(self) -> str:
        return "Fisher-KPP Problem(1800)"


class FisherDATA2000(FisherDATABase):
    """Fisher-KPP with data2000.mat initial conditions."""

    def __init__(self, ic_data_path: Optional[str] = None, data_root: str = DEFAULT_DATA_ROOT):
        if ic_data_path is None:
            ic_data_path = f"{data_root}/data2000.mat"
        super().__init__(
            ic_data_path=ic_data_path,
            gamma=-1030.0,
            lambda1=-0.064,
            lambda2=64/1.7
        )

    @property
    def name(self) -> str:
        return "Fisher-KPP Problem(2000)"
