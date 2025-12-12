import torch
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional, Union, Dict, Any
from scipy.io import loadmat
from configs.config import TrainingConfig


class PointGenerator(ABC):
    """Abstract base class for point generators."""

    @abstractmethod
    def generate_points(self, domain: List[Tuple[float, float]], n_points: int, dtype: torch.dtype) -> torch.Tensor:
        """Generate a specified number of points in the given domain."""
        pass


class GridPointGenerator(PointGenerator):
    """Uniform grid point generator for collocation points."""

    def generate_points(self, domain: List[Tuple[float, float]], n_points: int, dtype: torch.dtype) -> torch.Tensor:
        """Generate uniformly distributed grid points."""
        dim = len(domain)
        if dim == 0:
            return torch.empty((n_points, 0), dtype=dtype)
        elif dim == 1:
            x_min, x_max = domain[0]
            points = torch.linspace(x_min, x_max, n_points, dtype=dtype).view(-1, 1)
        else:
            # Calculate number of points per dimension
            points_per_dim = max(2, int(np.ceil(n_points ** (1/dim))))

            # Generate points for each dimension
            dim_points = []
            for d_min, d_max in domain:
                dim_points.append(torch.linspace(d_min, d_max, points_per_dim, dtype=dtype))

            # Create mesh grid
            mesh_grids = torch.meshgrid(*dim_points, indexing='ij')
            points = torch.stack([grid.flatten() for grid in mesh_grids], dim=1)

            # If generated points exceed required count, randomly sample
            if points.shape[0] > n_points:
                indices = torch.randperm(points.shape[0])[:n_points]
                points = points[indices]

        return points


class RandomPointGenerator(PointGenerator):
    """Random point generator using uniform distribution."""

    def generate_points(self, domain: List[Tuple[float, float]], n_points: int, dtype: torch.dtype) -> torch.Tensor:
        """Generate randomly distributed points."""
        dim = len(domain)
        points = torch.rand(n_points, dim, dtype=dtype)

        # Scale to domain range
        for i, (d_min, d_max) in enumerate(domain):
            points[:, i] = points[:, i] * (d_max - d_min) + d_min

        return points


class BoundaryPointGenerator(PointGenerator):
    """Boundary point generator for domain boundaries."""

    def generate_points(self, problem_type, domain: List[Tuple[float, float]], n_points: int, dtype: torch.dtype) -> torch.Tensor:
        """Generate points on domain boundaries."""
        dim = len(domain)

        if dim == 1:
            # 1D: Only two boundary points (endpoints)
            d_min, d_max = domain[0]
            n_left = n_points // 2
            n_right = n_points - n_left
            left_points = torch.full((n_left, 1), d_min, dtype=dtype)
            right_points = torch.full((n_right, 1), d_max, dtype=dtype)
            return torch.cat([left_points, right_points], dim=0)
        else:
            # Multi-dimensional: Generate points on each face
            boundary_points = []

            # Determine which dimensions are spatial (need boundary conditions)
            spatial_dims = list(range(dim))
            if problem_type == 'time-dependent':
                # Assume the last dimension is time
                spatial_dims = list(range(dim - 1))

            # Number of points allocated to each spatial boundary face
            points_per_face = max(1, n_points // (2 * len(spatial_dims))) if spatial_dims else 0

            for face_dim in spatial_dims:
                # Get domain bounds for other dimensions
                other_dims = [domain[i] for i in range(dim) if i != face_dim]

                if other_dims:
                    # Generate points in other dimensions
                    other_points = GridPointGenerator().generate_points(other_dims, points_per_face, dtype)

                    # Create points for left and right boundaries of current dimension
                    left_points = torch.zeros(points_per_face, dim, dtype=dtype)
                    right_points = torch.zeros(points_per_face, dim, dtype=dtype)

                    # Set boundary values for current dimension
                    left_points[:, face_dim] = domain[face_dim][0]
                    right_points[:, face_dim] = domain[face_dim][1]

                    # Fill in other dimensions (ensure left/right boundaries have same values in other dims)
                    other_idx = 0
                    for i in range(dim):
                        if i != face_dim:
                            left_points[:, i] = other_points[:, other_idx]
                            right_points[:, i] = other_points[:, other_idx]
                            other_idx += 1

                    # Add all points from one face first, then the other (important for periodic BCs)
                    boundary_points.append(left_points)
                    boundary_points.append(right_points)

            return torch.cat(boundary_points, dim=0) if boundary_points else torch.empty(0, dim, dtype=dtype)


class InitPointGenerator(PointGenerator):
    """Initial condition point generator for time-dependent problems."""

    def generate_points(self, problem_type, domain: List[Tuple[float, float]], n_points: int, dtype: torch.dtype) -> torch.Tensor:
        """Generate spatial points at initial time t=t_min."""
        dim = len(domain)

        if problem_type != 'time-dependent':
            raise ValueError("Initial points can only be generated for time-dependent problems (at least 2D).")

        # Spatial domain (excluding time dimension)
        spatial_domain = domain[:-1]
        spatial_points = GridPointGenerator().generate_points(spatial_domain, n_points, dtype)

        # Add time dimension set to t_min
        t_min = domain[-1][0]
        t_column = torch.full((spatial_points.shape[0], 1), t_min, dtype=dtype)

        return torch.cat([spatial_points, t_column], dim=1)


class MeshFilter:
    """Mesh point filter utilities."""

    @staticmethod
    def apply_filters(points: torch.Tensor, filters: List[Callable[[torch.Tensor], torch.Tensor]]) -> torch.Tensor:
        """Apply a series of filters to points."""
        for filter_func in filters:
            mask = filter_func(points)
            points = points[mask]
        return points

    @staticmethod
    def exclude_boundary_filter(tolerance: float = 1e-6):
        """Filter to exclude boundary points."""
        raise NotImplementedError("This method has not been implemented yet.")

    @staticmethod
    def custom_region_filter(region_func: Callable[[torch.Tensor], torch.Tensor]):
        """Custom region filter."""
        return region_func


class ExternalDataLoader:
    """Handles loading external data from .mat files.

    This class centralizes all scipy.io operations for the project,
    providing a clean interface for loading initial condition data
    and visualization reference data.
    """

    @staticmethod
    def load_initial_condition_data(
        file_path: str,
        n_points: int,
        input_dim: int,
        dtype: torch.dtype,
        domain: List[Tuple[float, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load initial condition data from a .mat file.

        Args:
            file_path: Path to the .mat file
            n_points: Number of points to sample/interpolate
            input_dim: Dimension of input space (e.g., 2 for x,t)
            dtype: PyTorch dtype for output tensors
            domain: List of (min, max) tuples for each dimension

        Returns:
            Tuple of (ic_points, ic_target):
                - ic_points: Tensor of shape (n_points, input_dim) with (x, t_min) coordinates
                - ic_target: Tensor of shape (n_points, 1) with u values at initial time

        Note:
            Returns empty tensors if file not found or has invalid format.
            Expects .mat file with keys 'x', 't', 'p' where p[t_idx] gives u values.
        """
        if not os.path.exists(file_path):
            print(f"Warning: IC data file not found at {file_path}. Using grid-based ICs only.")
            return (
                torch.empty(0, input_dim, dtype=dtype),
                torch.empty(0, 1, dtype=dtype)
            )

        print(f"Loading initial condition data from {file_path}...")
        data = loadmat(file_path)

        # Validate required keys
        if 'x' not in data or 'p' not in data:
            print(f"Error: .mat file '{file_path}' must contain 'x' and 'p' keys. "
                  f"Found: {list(data.keys())}. Using grid-based ICs only.")
            return (
                torch.empty(0, input_dim, dtype=dtype),
                torch.empty(0, 1, dtype=dtype)
            )

        x_arr = data['x'].squeeze()
        t_arr = data['t'].squeeze()
        p_arr = data['p']

        # Find t=0 index
        t0_idx = np.where(t_arr == 0)[0]
        if len(t0_idx) == 0:
            raise ValueError("No t=0 found in mat file.")
        t0_idx = t0_idx[0]

        x_data_full = torch.from_numpy(x_arr).to(dtype).view(-1, 1)
        u_data_full = torch.from_numpy(p_arr[t0_idx]).to(dtype).view(-1, 1)

        n_data_total = x_data_full.shape[0]

        if n_points > n_data_total:
            # Interpolate to get more points
            print(f"Warning: Requested {n_points} data points, but only {n_data_total} are available. Interpolating.")
            x_full_np = x_data_full.view(-1).cpu().numpy()
            u_full_np = u_data_full.view(-1).cpu().numpy()
            x_interp = np.linspace(x_full_np.min(), x_full_np.max(), n_points)
            u_interp = np.interp(x_interp, x_full_np, u_full_np)
            x_data = torch.from_numpy(x_interp).to(dtype).view(-1, 1)
            u_data = torch.from_numpy(u_interp).to(dtype).view(-1, 1)
        elif n_points < n_data_total:
            # Random sampling
            print(f"Randomly sampling {n_points} points from {n_data_total} available IC data points.")
            indices = torch.randperm(n_data_total)[:n_points]
            x_data = x_data_full[indices]
            u_data = u_data_full[indices]
        else:
            print(f"Using all {n_data_total} available IC data points.")
            x_data = x_data_full
            u_data = u_data_full

        # Build IC points tensor with (x, t_min) coordinates
        if x_data.dim() == 1:
            x_data = x_data.view(-1, 1)

        # Get t_min from domain (assuming time is last dimension for time-dependent problems)
        t_min = domain[-1][0] if len(domain) > 1 else 0.0
        t_data = torch.full_like(x_data, t_min)
        ic_points = torch.cat([x_data, t_data], dim=1)

        ic_target = u_data
        if ic_target.dim() == 1:
            ic_target = ic_target.view(-1, 1)

        return ic_points, ic_target

    @staticmethod
    def load_visualization_data(file_path: str) -> Dict[str, Any]:
        """
        Load .mat file data for visualization purposes.

        Args:
            file_path: Path to the .mat file

        Returns:
            Dictionary containing the loaded data with keys typically including
            'x', 't', 'p' (and potentially others)

        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Visualization data file not found: {file_path}")
        return loadmat(file_path)

class MeshGenerator:
    """Unified mesh generator for all point types."""

    def __init__(self,
                 collocation_generator: PointGenerator = None,
                 boundary_generator: PointGenerator = None,
                 initial_generator: PointGenerator = None):
        """
        Initialize the mesh generator.

        Args:
            collocation_generator: Generator for collocation (interior) points
            boundary_generator: Generator for boundary condition points
            initial_generator: Generator for initial condition points
        """
        self.collocation_generator = collocation_generator or GridPointGenerator()
        self.boundary_generator = boundary_generator or BoundaryPointGenerator()
        self.initial_generator = initial_generator or InitPointGenerator()
    
    @staticmethod
    def get_points(problem,
                   train_cfg: TrainingConfig,
                   dtype: torch.dtype = torch.float64,
                   collocation_filters: Optional[List[Callable]] = None,
                   boundary_filters: Optional[List[Callable]] = None,
                   initial_filters: Optional[List[Callable]] = None,
                   custom_generators: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unified point generation interface.

        Args:
            problem: Problem instance
            train_cfg: Training configuration
            dtype: Data type for tensors
            collocation_filters: Filters for collocation points
            boundary_filters: Filters for boundary points
            initial_filters: Filters for initial points
            custom_generators: Custom generator dictionary

        Returns:
            (collocation_points, boundary_points, initial_points, ic_data_points, ic_data_target)

            The last two tensors contain external IC data if available (from .mat files),
            otherwise empty tensors. This allows the Solver to be agnostic about data source.
        """
        # Normalize domain configuration
        domain_cfg = problem.domain()
        if isinstance(domain_cfg[0], (int, float)):
            domain = [domain_cfg]  # Convert 1D to standard format
        else:
            domain = domain_cfg

        print(f"Generating mesh for {len(domain)}D problem with domain: {domain}")

        # Use custom generators or defaults
        generators = custom_generators or {}
        mesh_gen = MeshGenerator(
            collocation_generator=generators.get('collocation', GridPointGenerator()),
            boundary_generator=generators.get('boundary', BoundaryPointGenerator()),
            initial_generator=generators.get('initial', InitPointGenerator())
        )

        # Generate collocation points
        if train_cfg.n_colloc > 0:
            colloc = mesh_gen.collocation_generator.generate_points(domain, train_cfg.n_colloc, dtype)
            if collocation_filters:
                colloc = MeshFilter.apply_filters(colloc, collocation_filters)
        else:
            colloc = torch.empty(0, len(domain), dtype=dtype)

        # Generate boundary points
        if train_cfg.n_bc > 0:
            bc = mesh_gen.boundary_generator.generate_points(problem.problem_type, domain, train_cfg.n_bc, dtype)
            if boundary_filters:
                bc = MeshFilter.apply_filters(bc, boundary_filters)
        else:
            bc = torch.empty(0, len(domain), dtype=dtype)

        # Generate initial points
        if train_cfg.n_init > 0:
            # For time-dependent problems, initial points are at t=t_min
            init = mesh_gen.initial_generator.generate_points(problem.problem_type, domain, train_cfg.n_init, dtype)
            if initial_filters:
                init = MeshFilter.apply_filters(init, initial_filters)
        else:
            init = torch.empty(0, len(domain), dtype=dtype)

        # Load external IC data if problem has ic_data_path and n_ic_data > 0
        ic_path = getattr(problem, 'ic_data_path', None)
        if ic_path and train_cfg.n_ic_data > 0:
            ic_data_points, ic_data_target = ExternalDataLoader.load_initial_condition_data(
                ic_path, train_cfg.n_ic_data, len(domain), dtype, domain
            )
        else:
            ic_data_points = torch.empty(0, len(domain), dtype=dtype)
            ic_data_target = torch.empty(0, 1, dtype=dtype)

        return colloc, bc, init, ic_data_points, ic_data_target