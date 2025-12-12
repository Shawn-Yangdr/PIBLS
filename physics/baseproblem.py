from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import List, Tuple, Union, Optional, Callable


@dataclass
class PDETerm:
    """
    Describes a single linear differential operator term in the PDE.

    For a term like c(x) * (d^n u / dx_i^n), this captures:
    - derivative_order: n (order of differentiation)
    - var_index: i (which variable to differentiate with respect to)
    - weight_power: power applied to weights in derivative computation
    - coefficient: c(x) - can be a float constant or callable for variable coefficients

    Example:
        # For -0.01 * u_xx (diffusion term):
        PDETerm(derivative_order=2, var_index=0, weight_power=2, coefficient=-0.01)

        # For (1+x) * u_x (variable coefficient advection):
        PDETerm(derivative_order=1, var_index=0, weight_power=1, coefficient=lambda x: 1+x[:,0:1])
    """
    derivative_order: int
    var_index: int
    weight_power: int
    coefficient: Union[float, callable] = 1.0


@dataclass
class NonlinearTerm:
    """
    Describes a single nonlinear term in the PDE.

    Supported term types:
    - 'sin(u)': Applies sin() to the solution, e.g., beta * sin(u) in Helmholtz
    - 'u*ux': Product of solution and its derivative, e.g., u * u_x in Burgers
    - 'u^2': Square of solution, e.g., lambda * u^2 in Fisher-KPP

    Attributes:
        term_type: String identifier for the nonlinear operation
        var_index: Variable index for derivatives (e.g., x=0, t=1 for u*ux)
        coefficient: Multiplicative coefficient for the term
    """
    term_type: str  # e.g., 'u*ux', 'u^2', 'sin(u)'
    var_index: int  # Primary variable index for derivatives like u_x
    coefficient: float = 1.0


@dataclass
class ResidualTerm:
    """
    Encapsulates a complete residual constraint for the solver.

    Each residual represents a constraint of the form: L(u) = target
    where L is a linear differential operator (or identity for direct evaluation).

    The solver builds the linear system by computing:
    - H matrix: basis functions for the operator L applied to BLS features
    - K vector: target values from target_func

    Attributes:
        name: Identifier for the residual (e.g., 'pde', 'bc', 'ic')
        points: Tensor of evaluation points for this constraint
        target_func: Function that computes target values given denormalized points
        operator_terms: List of PDETerms defining L(u). If None, evaluates u directly.
        is_periodic: If True, points are paired (first half = left, second half = right)
                    and constraint becomes u(left) - u(right) = 0

    Example:
        # PDE residual: u_xx + u_yy = f(x,y)
        ResidualTerm(
            name='pde',
            points=colloc_points,
            target_func=source_term_func,
            operator_terms=[PDETerm(2, 0, 2), PDETerm(2, 1, 2)]
        )

        # Dirichlet BC: u = g(x) on boundary
        ResidualTerm(
            name='bc',
            points=boundary_points,
            target_func=boundary_func,
            operator_terms=None  # Direct evaluation of u
        )

        # Periodic BC: u(x_left, t) = u(x_right, t)
        ResidualTerm(
            name='bc',
            points=torch.cat([left_points, right_points]),
            target_func=lambda x: torch.zeros(...),  # Not used when is_periodic=True
            operator_terms=None,
            is_periodic=True
        )
    """
    name: str  # Residual identifier, e.g., 'pde', 'ic_u', 'ic_u_deriv'
    points: torch.Tensor  # Points where this residual is evaluated
    target_func: Callable[[torch.Tensor], torch.Tensor]  # Function to compute target values
    operator_terms: Optional[List[PDETerm]] = None  # Linear operator to apply; None means evaluate u directly
    is_periodic: bool = False  # If True, handles periodic BC (points are paired)

class BaseProblem(ABC):
    """
    Abstract Base Class for PDE problem definitions.

    Subclasses must implement:
    - name: Problem identifier
    - input_dim: Spatial/temporal dimension (1 for ODE, 2+ for PDE)
    - problem_type: 'spatial' or 'time-dependent'
    - domain(): Returns domain bounds
    - exact_solution(): Analytical solution (for error computation)
    - pde_source_term(): Right-hand side f(x) in L(u) = f(x)

    Optional overrides:
    - linear_terms(): Define differential operators
    - nonlinear_terms(): Define nonlinear terms
    - boundary_conditions(): Custom BC values (defaults to exact solution)
    - initial_conditions(): Custom IC values (defaults to exact solution)
    - get_residual_terms(): Custom residual assembly (for Neumann BCs, etc.)
    - get_boundary_filters(): Filters for boundary point selection
    """
    @abstractmethod
    def __init__(self):
        self.pde_target = None
        self.bc_target = None
        self.init_target = None

    def get_residual_terms(self, colloc: torch.Tensor, bc: torch.Tensor, init: torch.Tensor) -> List[ResidualTerm]:
        """
        Generate all constraint residuals for this problem.

        This is the core method that defines the optimization problem.
        Override this method for custom boundary conditions (e.g., Neumann, periodic).

        Args:
            colloc: Collocation points for PDE residual
            bc: Boundary points for BC residual
            init: Initial condition points (for time-dependent problems)

        Returns:
            List of ResidualTerm objects defining all constraints
        """
        terms = []

        # 1. PDE residual (applies to all problems)
        if colloc.numel() > 0:
            # Note: Nonlinear terms are handled separately by the Trainer
            terms.append(ResidualTerm(
                name='pde',
                points=colloc,
                target_func=self.pde_source_term,
                operator_terms=self.linear_terms()
            ))

        # 2. Boundary condition residual
        if bc.numel() > 0:
            terms.append(ResidualTerm(
                name='bc',
                points=bc,
                target_func=self.boundary_conditions,
                operator_terms=None  # operator_terms=None means direct evaluation of u
            ))

        # 3. Initial condition residual (u(x, 0))
        if init.numel() > 0:
            terms.append(ResidualTerm(
                name='ic',
                points=init,
                target_func=self.initial_conditions,
                operator_terms=None  # operator_terms=None means direct evaluation of u
            ))

        return terms
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the equation."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """The dimension of the problem's input domain (e.g., 1 for 1D, 2 for 1D+time)."""
        pass

    @property
    @abstractmethod
    def problem_type(self) -> str:
        """The type of the problem, e.g., 'spatial' or 'time-dependent'."""
        pass

    @property
    def boundary_type(self) -> str:
        """The type of boundary condition, e.g., 'dirichlet', 'periodic'."""
        return 'dirichlet'

    @abstractmethod
    def domain(self) -> Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]:
        """The problem domain, e.g., (0.0, 1.0) or ((0, 5), (0, 10))."""
        pass

    @abstractmethod
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """The exact solution u(x)."""
        pass

    @abstractmethod
    def pde_source_term(self, x: torch.Tensor) -> torch.Tensor:
        """The source term f(x) for the PDE L(u) = f(x)."""
        pass
        
    def linear_terms(self) -> List[PDETerm]:
        """A list of linear terms in the PDE."""
        return []
        
    def nonlinear_terms(self) -> List[NonlinearTerm]:
        """A list of nonlinear terms in the PDE."""
        return []
        
    def boundary_conditions(self, x_bc: torch.Tensor) -> torch.Tensor:
        """Defines the Dirichlet boundary condition values."""
        return self.exact_solution(x_bc)

    def get_boundary_filters(self):
        """
        Get problem-specific boundary point filters.

        Override this to filter boundary points, e.g., for inflow-only BCs.

        Returns:
            List of filter functions that take points tensor and return boolean mask
        """
        return []

    def initial_conditions(self, x_init: torch.Tensor) -> torch.Tensor:
        """
        Define initial condition values u(x, t=0).

        Default implementation uses the exact solution.
        Override for problems with different ICs or no analytical solution.
        """
        return self.exact_solution(x_init)

    def initial_derivative_conditions(self, x_init: torch.Tensor) -> torch.Tensor:
        """
        Define initial derivative condition values u_t(x, t=0).

        Used for second-order-in-time problems (e.g., wave equation).
        Default returns zeros; override if needed.
        """
        return torch.zeros_like(x_init[:, 0:1])

