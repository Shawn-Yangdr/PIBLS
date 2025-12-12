"""
Problem registry for PIBLS.

Maps problem names to their corresponding classes for CLI lookup.
"""
from physics.benchmarks import (
    TC1, TC2, TC3, TC4, TC5, TC6, TC7, TC8,
    TC9, TC10, TC11
)
from physics.applications import (
    FisherDATA1000, FisherDATA1200, FisherDATA1400,
    FisherDATA1600, FisherDATA1800, FisherDATA2000
)


# Problem registry - maps CLI names to problem classes
PROBLEM_REGISTRY = {
    # Benchmark test cases
    "TC1": TC1,
    "TC2": TC2,
    "TC3": TC3,
    "TC4": TC4,
    "TC5": TC5,
    "TC6": TC6,
    "TC7": TC7,
    "TC8": TC8,
    "TC9": TC9,
    "TC10": TC10,
    "TC11": TC11,
    # Application problems
    "Fisher1000": FisherDATA1000,
    "Fisher1200": FisherDATA1200,
    "Fisher1400": FisherDATA1400,
    "Fisher1600": FisherDATA1600,
    "Fisher1800": FisherDATA1800,
    "Fisher2000": FisherDATA2000,
}


def get_problem_class(name: str):
    """Get problem class by name. Returns None if not found."""
    return PROBLEM_REGISTRY.get(name)


def find_problem_class(problem_name: str):
    """
    Find problem class by problem.name attribute (used for model loading).

    Args:
        problem_name: The problem's .name attribute (e.g., "TC-1 Equation")

    Returns:
        Problem class or None if not found.
    """
    for key, cls in PROBLEM_REGISTRY.items():
        if cls().name == problem_name:
            return cls
    return None


def list_problems():
    """Return list of available problem names."""
    return list(PROBLEM_REGISTRY.keys())
