# benchmarks.py
"""
Benchmark test cases (TC1-TC11) for validating the PIBLS solver.

These problems have known analytical solutions and cover:
- 1D/2D spatial problems (TC1-TC6)
- Time-dependent advection (TC7-TC8)
- Nonlinear ODEs/PDEs (TC9-TC11)

Reference: Standard benchmark problems for physics-informed methods.
"""
from abc import ABC, abstractmethod
import torch
from typing import List, Tuple, Union, Optional

from physics.baseproblem import BaseProblem, PDETerm, NonlinearTerm, ResidualTerm

class TC1(BaseProblem):
    """
    u_x = R, x \in (0, 1)
    u^hat = sin(2πx)cos(4πx) + 1
    R = 2πcos(2πx)cos(4πx) - 4πsin(2πx)sin(4πx)
    """
    def __init__(self):
        super().__init__()
    
    @property
    def input_dim(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "TC-1 Equation"

    @property
    def problem_type(self) -> str:
        return 'spatial'

    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * torch.pi * x) * torch.cos(4 * torch.pi * x) + 1
    
    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.pi * torch.cos(2 * torch.pi * x) * torch.cos(4 * torch.pi * x) \
                - 4 * torch.pi * torch.sin(2 * torch.pi * x) * torch.sin(4 * torch.pi * x)
    

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=1, var_index=0, weight_power=1)
        ]

class TC2(BaseProblem):
    """
    u_xx = R, x \in (0, 1)
    u^hat = sin(πx/2)cos(2πx) + 1
    R = -17/4π^2sin(πx/2)cos(2πx) - 2π^2cos(πx/2)sin(2πx)
    """
    def __init__(self):
        super().__init__()

    @property 
    def input_dim(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "TC-2 Equation"
    
    @property
    def problem_type(self) -> str:
        return 'spatial'
    
    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x / 2) * torch.cos(2 * torch.pi * x) + 1
    
    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        return - 17 / 4 * torch.pi**2 * torch.sin(torch.pi * x / 2) * torch.cos(2 * torch.pi * x) \
                - 2 * torch.pi**2 * torch.cos(torch.pi * x / 2) * torch.sin(2 * torch.pi * x)

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=2, var_index=0, weight_power=2)
        ]

class TC3(BaseProblem):
    """
    u_x - v * u_xx = R, x \in (0, 1), v = 0.2
    u^hat = e^(x/v) - 1 / (e^(1/v) - 1)
    R = 0
    """
    def __init__(self, v: float = 0.2):
        super().__init__()
        self.v = torch.tensor(v)
    
    @property
    def input_dim(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "TC-3 Equation"
    
    @property
    def problem_type(self) -> str:
        return 'spatial'

    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.exp(x / self.v) - 1) / (torch.exp(1 / self.v) - 1) 
    
    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=1, var_index=0, weight_power=1),
            PDETerm(derivative_order=2, var_index=0, weight_power=2, coefficient=-self.v)
        ]

class TC4(BaseProblem):
    """
    au_x + bu_y = R, (x, y) \in (-1, 1)x(-1, 1), a=1,b=1/2
    u^hat = 1/2cos(πx)sin(πy)
    R = -π/2sin(πx)sin(πy) + π/4cos(πx)cos(πy)
    """
    def __init__(self, a: float = 1.0, b : float = 0.5):
        super().__init__()
        self.a = a
        self.b = b
    
    @property
    def input_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "TC-4 Equation"
    
    @property
    def problem_type(self) -> str:
        return 'spatial'

    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((-1.0, 1.0), (-1.0, 1.0))
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / 2 * torch.cos(torch.pi * x[:,0:1]) * torch.sin(torch.pi * x[:,1:2])
    
    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        u_x = - torch.pi / 2 * torch.sin(torch.pi * x[:,0:1]) * torch.sin(torch.pi * x[:,1:2]) 
        u_y = torch.pi / 2 * torch.cos(torch.pi * x[:,0:1]) * torch.cos(torch.pi * x[:,1:2])
        return self.a * u_x + self.b * u_y

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=1, var_index=0, weight_power=1, coefficient=self.a),
            PDETerm(derivative_order=1, var_index=1, weight_power=1, coefficient=self.b)    
        ]

class TC5(BaseProblem):
    """
    u_xx + u_yy = R, (x, y) \in (0, 1)x(0, 1)
    u^hat = 1/2 + e^(-2x^2 - 4y^2)
    R = (16x^2 - 4)e^(-2x^2 - 4y^2) + (64y^2 - 8)e^(-2x^2 - 4y^2)
    """
    def __init__(self):
        super().__init__()

    @property
    def input_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "TC-5 Equation"
    
    @property
    def problem_type(self) -> str:
        return 'spatial'
    
    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((0, 1.0), (0, 1.0))
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / 2 + torch.exp(-2 * x[:,0:1]**2 - 4 * x[:,1:2]**2)
    
    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        u_xx = (16 * x[:,0:1]**2 - 4) * torch.exp(-2 * x[:,0:1]**2 - 4 * x[:,1:2]**2)
        u_yy = (64 * x[:,1:2]**2 - 8) * torch.exp(-2 * x[:,0:1]**2 - 4 * x[:,1:2]**2)
        return u_xx + u_yy

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=2, var_index=0, weight_power=2),
            PDETerm(derivative_order=2, var_index=1, weight_power=2)    
        ]

class TC6(BaseProblem):
    """
    u_xx + u_yy = R, (x, y) \in (0, 1)x(0, 1)
    u^hat = 1/2 + e^(-(x-0.6)^2 - (y-0.6)^2)
    R = (4(x-0.6)^2 + 4(y-0.6)^2 - 4)e^(-(x-0.6)^2 - (y-0.6)^2)
    """
    def __init__(self):
        super().__init__()

    @property
    def input_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "TC-6 Equation"
    
    @property
    def problem_type(self) -> str:
        return 'spatial'
    
    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((0, 1.0), (0, 1.0))
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / 2 + torch.exp(- (x[:,0:1] - 0.6)**2 - (x[:,1:2] - 0.6)**2)
    
    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        u_xx = ((- 2 * x[:,0:1] + 1.2)**2 - 2) * torch.exp(- (x[:,0:1] - 0.6)**2 - (x[:,1:2] - 0.6)**2)
        u_yy = ((- 2 * x[:,1:2] + 1.2)**2 - 2) * torch.exp(- (x[:,0:1] - 0.6)**2 - (x[:,1:2] - 0.6)**2)
        return u_xx + u_yy

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=2, var_index=0, weight_power=2),
            PDETerm(derivative_order=2, var_index=1, weight_power=2)    
        ]

class TC7(BaseProblem):
    """
    u_t + a(x)u_x = 0, (x, t) \in (-1,1)x(0,0.5)
    u(x,0) = sin(πx), x \in(-1, 1)
    u^hat = sin(π(x - t))
    a(x) = 1
    """
    def __init__(self):
        super().__init__()

    @property
    def input_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "TC-7 Equation"

    @property
    def problem_type(self) -> str:
        return 'time-dependent'

    @property
    def boundary_type(self) -> str:
        return 'periodic'

    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((-1.0, 1.0), (0.0, 0.5))

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * (x[:,0:1] - x[:,1:2]))

    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:,0:1])

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=1, var_index=1, weight_power=1), # u_t
            PDETerm(derivative_order=1, var_index=0, weight_power=1, coefficient=1.0) # u_x
        ]

    def get_residual_terms(self, colloc: torch.Tensor, bc: torch.Tensor, init: torch.Tensor) -> List[ResidualTerm]:
        """Override to implement periodic boundary conditions."""
        terms = []

        # 1. PDE residual
        if colloc.numel() > 0:
            terms.append(ResidualTerm(
                name='pde',
                points=colloc,
                target_func=self.pde_source_term,
                operator_terms=self.linear_terms()
            ))

        # 2. Periodic boundary condition: u(x_left, t) = u(x_right, t)
        # BC points come as pairs: first half are left boundary, second half are right
        if bc.numel() > 0:
            terms.append(ResidualTerm(
                name='bc',
                points=bc,
                target_func=lambda x: torch.zeros_like(x[:, 0:1]),  # Target is 0 (not used)
                operator_terms=None,
                is_periodic=True  # Signal to solver to compute H_left - H_right
            ))

        # 3. Initial condition
        if init.numel() > 0:
            terms.append(ResidualTerm(
                name='ic',
                points=init,
                target_func=self.initial_conditions,
                operator_terms=None
            ))

        return terms

class TC8(BaseProblem):
    """
    u_t + a(x)u_x = 0, (x, t) \in (-1,1)x(0,0.5)
    u(x,0) = sin(πx), x \in(-1, 1)
    u^hat = sin(π((1+x)e^(-t) - 1))
    a(x) = 1 + x
    """
    def __init__(self):
        super().__init__()
    
    @property
    def input_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "TC-8 Equation"
    
    @property
    def problem_type(self) -> str:
        return 'time-dependent'
    
    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((-1.0, 1.0), (0.0, 0.5))
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * ((1 + x[:,0:1]) * torch.exp(-x[:,1:2]) - 1))
    
    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:,0:1])
    
    def variable_coefficient(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the variable coefficient (1 + x) for the u_x term."""
        return 1.0 + x[:,0:1]
    
    def boundary_conditions(self, x_bc):
        return super().boundary_conditions(x_bc)
    
    def get_boundary_filters(self):
        """Get TC8-specific boundary filters (left boundary only for inflow)."""
        def left_only_filter(points):
            return torch.abs(points[:, 0] + 1.0) < 1e-5

        return [left_only_filter]

    def linear_terms(self) -> List[PDETerm]:
        # For the equation u_t + (1 + x) * u_x = 0
        # We need to handle the variable coefficient (1 + x) separately
        # This returns the basic linear structure
        return [
            PDETerm(derivative_order=1, var_index=1, weight_power=1), # u_t
            PDETerm(derivative_order=1, var_index=0, weight_power=1, 
                   coefficient=self.variable_coefficient) # (1+x)*u_x
        ]

class TC9(BaseProblem):
    """
    1D Nonlinear Helmholtz equation:
    u_xx - lambda*u + beta*sin(u) = R, x in (0, 8)

    Parameters: lambda=50, beta=10
    Exact: u = sin(3*pi*x + 3*pi/20)*cos(4*pi*x - 2*pi/5) + 1.5 + x/10

    This problem tests nonlinear sin(u) term handling.
    """
    def __init__(self, lambda_=50.0, beta=10.0):
        super().__init__()
        self.lambda_ = lambda_
        self.beta = beta
    
    @property
    def name(self) -> str:
        return "TC-9 Equation"
    
    @property
    def input_dim(self) -> int:
        return 1
    
    @property
    def problem_type(self) -> str:
        return 'spatial'

    def domain(self) -> Tuple[float, float]:
        return (0.0, 8.0)

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.sin(3 * torch.pi * x + 3 * torch.pi / 20) *
                torch.cos(4 * torch.pi * x - 2 * torch.pi / 5) +
                1.5 + x / 10)

    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        u = self.exact_solution(x)
        u_xx = (-25 * torch.pi**2 * torch.sin(3 * torch.pi * x + 3 * torch.pi / 20) *
                torch.cos(4 * torch.pi * x - 2 * torch.pi / 5) -
                24 * torch.pi**2 * torch.cos(3 * torch.pi * x + 3 * torch.pi / 20) *
                torch.sin(4 * torch.pi * x - 2 * torch.pi / 5))
        return u_xx - self.lambda_ * u + self.beta * torch.sin(u)

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=2, var_index=0, weight_power=2),
            PDETerm(derivative_order=0, var_index=0, weight_power=0, coefficient=-self.lambda_),
        ]

    def nonlinear_terms(self) -> List[NonlinearTerm]:
        return [
            NonlinearTerm(term_type='sin(u)', var_index=0, coefficient=self.beta)
        ]

class TC10(BaseProblem):
    """
    Nonlinear spring (Duffing-like) ODE:
    u_tt + omega^2*u + alpha*sin(u) = R, t in (0, 2.5)

    Parameters: omega=2, alpha=0.1
    Exact: u = t * sin(t)
    Initial conditions: u(0) = 0, u'(0) = 0

    This problem tests second-order-in-time with derivative ICs.
    """
    def __init__(self, omega=2.0, alpha=0.1):
        super().__init__()
        self.omega = omega
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "TC-10 Equation"

    @property
    def input_dim(self) -> int:
        return 1

    @property
    def problem_type(self) -> str:
        return 'time-dependent'

    def domain(self) -> Tuple[float, float]:
        return (0.0, 2.5)

    def exact_solution(self, t: torch.Tensor) -> torch.Tensor:
        return t * torch.sin(t)

    def pde_source_term(self, t: torch.Tensor) -> torch.Tensor:
        u = self.exact_solution(t)
        u_tt = 2 * torch.cos(t) - t * torch.sin(t)
        return u_tt + self.omega**2 * u + self.alpha * torch.sin(u)

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=2, var_index=0, weight_power=2),  # u_tt
            PDETerm(derivative_order=0, var_index=0, weight_power=0, coefficient=self.omega**2),  # omega^2 * u
        ]

    def nonlinear_terms(self) -> List[NonlinearTerm]:
        return [
            NonlinearTerm(term_type='sin(u)', var_index=0, coefficient=self.alpha)
        ]

    def get_residual_terms(self, colloc: torch.Tensor, bc: torch.Tensor, init: torch.Tensor) -> List[ResidualTerm]:
        # First call parent method to get PDE and u(0) residuals
        terms = super().get_residual_terms(colloc, bc, init)

        # Add residual for initial derivative condition u'(0) = 0
        if init.numel() > 0:
            # Define the u'(t) operator
            du_dt_operator = [PDETerm(derivative_order=1, var_index=0, weight_power=1)]

            terms.append(ResidualTerm(
                name='ic_u_deriv',
                points=init,
                target_func=self.initial_derivative_conditions,
                operator_terms=du_dt_operator
            ))

        return terms

    def initial_derivative_conditions(self, t: torch.Tensor) -> torch.Tensor:
        # u'(t) = sin(t) + t*cos(t), which equals 0 at t=0
        return torch.zeros_like(t)

class TC11(BaseProblem):
    """
    Viscous Burgers equation:
    u_t + u * u_x - v * u_xx = R, (x, t) in (0, 1) x (0, 0.25)

    Parameters: v=0.01 (viscosity)
    Exact: u = (1 + x/10)(1 + t/10) * [2cos(pi*x + 2pi/5) + 1.5cos(2pi*x - 3pi/5)]
                                    * [2cos(pi*t + 2pi/5) + 1.5cos(2pi*t - 3pi/5)]

    This problem tests the u*u_x nonlinear convection term.
    """
    def __init__(self, v=0.01):
        super().__init__()
        self.v = v

    @property
    def name(self) -> str:
        return "TC-11 Equation"
    
    @property
    def input_dim(self) -> int:
        return 2

    @property
    def problem_type(self) -> str:
        return 'time-dependent'

    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((0.0, 1.0), (0.0, 0.25))  # (x_domain, t_domain)

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        solution = (1 + x[:, 0:1] / 10) * (1 + x[:, 1:2] / 10)  \
        * (2 * torch.cos(torch.pi * x[:, 0:1] + 2 * torch.pi / 5) + \
            1.5 * torch.cos(2 * torch.pi * x[:, 0:1] - 3 * torch.pi / 5)) \
        * (2 * torch.cos(torch.pi * x[:, 1:2] + 2 * torch.pi / 5) + \
            1.5 * torch.cos(2 * torch.pi * x[:, 1:2] - 3 * torch.pi / 5))
        return solution

    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        space = x[:, 0]
        times = x[:, 1]

        A = 1 + space / 10
        B = 1 + times / 10

        x_angle1 = torch.pi * space + 2 * torch.pi / 5
        x_angle2 = 2 * torch.pi * space - 3 * torch.pi / 5
        t_angle1 = torch.pi * times + 2 * torch.pi / 5
        t_angle2 = 2 * torch.pi * times - 3 * torch.pi / 5

        C = 2 * torch.cos(x_angle1) + 1.5 * torch.cos(x_angle2)
        D = 2 * torch.cos(t_angle1) + 1.5 * torch.cos(t_angle2)

        dC_dx = -2 * torch.pi * torch.sin(x_angle1) - 3 * torch.pi * torch.sin(x_angle2)
        dC_dxx = -2 * torch.pi**2 * torch.cos(x_angle1) - 6 * torch.pi**2 * torch.cos(x_angle2)
        dD_dt = -2 * torch.pi * torch.sin(t_angle1) - 3 * torch.pi * torch.sin(t_angle2)

        u = A * B * C * D

        u_x = (1/10) * B * C * D + A * B * dC_dx * D
        u_t = A * (1/10) * C * D + A * B * C * dD_dt
        u_xx = (1/5) * B * dC_dx * D + A * B * dC_dxx * D

        return u_t + u * u_x - self.v * u_xx

    def linear_terms(self) -> List[PDETerm]:
        return [
            PDETerm(derivative_order=1, var_index=1, weight_power=1), # u_t
            PDETerm(derivative_order=2, var_index=0, weight_power=2, coefficient=-self.v), # -v*u_xx
        ]

    def nonlinear_terms(self) -> List[NonlinearTerm]:
        return [
            NonlinearTerm(term_type='u*ux', var_index=0, coefficient=1.0) # u*u_x
        ]
