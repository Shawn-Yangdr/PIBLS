"""
Broad Learning System (BLS) implementation.

This module provides the core BLS architecture with:
- NodeGenerator: Random feature generation with optional orthogonalization
- BLS: Standard BLS classifier/regressor (not used in PIBLS, kept for reference)

The BLS architecture generates features through:
1. Feature (mapping) layer: Random projections with activation
2. Enhancement layer: Additional random projections on feature outputs

Key insight: BLS uses random, fixed weights for feature extraction,
then learns only the output layer weights via least squares.

Note: DerivativeCalculator has been moved to trainer/derivatives.py
to maintain separation of concerns between model and physics.
"""
from __future__ import division
import numpy as np
import torch
from typing import Callable


class NodeGenerator:
    """
    Generates random feature nodes for the BLS network.

    This class handles:
    1. Random weight/bias generation with optional orthogonalization
    2. Forward transformation through the node layer
    3. Weight extraction for derivative calculations

    Attributes:
        weight_list: List of weight matrices for each node group
        bias_list: List of bias values for each node group
        activation_function: The activation function to apply
        whiten: Whether to orthogonalize weights (improves conditioning)
        random_range: Range for uniform random initialization [-range, range]
    """

    def __init__(self, whiten=False, random_range=2.0):
        """
        Initialize the node generator.

        Args:
            whiten: If True, orthogonalize weight matrices using Gram-Schmidt
            random_range: Weights initialized uniformly in [-range, range]
        """
        self.weight_list = []
        self.bias_list = []
        self.activation_function = None
        self.activation_name = None  # Store name for serialization
        self.whiten = whiten
        self.random_range = random_range

    def sigmoid(self, data):
        """Sigmoid activation: 1 / (1 + exp(-x))"""
        return 1.0 / (1 + np.exp(-data))

    def linear(self, data):
        """Linear (identity) activation."""
        return data

    def tanh(self, data):
        """Hyperbolic tangent activation."""
        return np.tanh(data)

    def relu(self, data):
        """Rectified Linear Unit activation: max(0, x)"""
        return np.maximum(data, 0)

    def sin(self, data):
        """Sinusoidal activation."""
        return np.sin(data)

    def orth(self, weight_matrix):
        """
        Orthogonalize weight matrix columns using Gram-Schmidt process.

        This improves numerical conditioning of the feature matrix,
        which is important for the least squares solve.

        Args:
            weight_matrix: Matrix to orthogonalize (modified in-place)

        Returns:
            Orthogonalized weight matrix with unit-norm columns
        """
        for i in range(0, weight_matrix.shape[1]):
            current_column = np.mat(weight_matrix[:, i].copy()).T
            orthogonal_sum = 0
            for j in range(i):
                previous_column = np.mat(weight_matrix[:, j].copy()).T
                orthogonal_sum += (current_column.T.dot(previous_column))[0, 0] * previous_column
            current_column -= orthogonal_sum
            current_column = current_column / np.sqrt(current_column.T.dot(current_column))
            weight_matrix[:, i] = np.ravel(current_column)
        return weight_matrix

    def generate_weights_and_biases(self, shape, num_generations):
        """
        Generator that yields (weight_matrix, bias) pairs.

        Args:
            shape: Shape of weight matrix (input_dim, hidden_dim)
            num_generations: Number of weight/bias pairs to generate

        Yields:
            Tuple of (weight_matrix, bias_scalar)
        """
        for _ in range(num_generations):
            weight_matrix = (2 * self.random_range) * np.random.random(size=shape) - self.random_range
            if self.whiten:
                weight_matrix = self.orth(weight_matrix)
            bias = (2 * self.random_range) * np.random.random() - self.random_range
            yield (weight_matrix, bias)

    def generator_nodes(self, data, num_generations, hidden_dim, activation_name):
        """
        Generate feature nodes by creating random projections of input data.

        This is the main method for creating the BLS feature layer.

        Args:
            data: Input data array of shape (n_samples, input_dim)
            num_generations: Number of node groups to generate
            hidden_dim: Output dimension for each node group
            activation_name: Name of activation function to use

        Returns:
            Transformed data after applying all node groups and activation
            Shape: (n_samples, num_generations * hidden_dim)
        """
        # Generate and store weights/biases
        self.weight_list = [elem[0] for elem in self.generate_weights_and_biases((data.shape[1], hidden_dim), num_generations)]
        self.bias_list = [elem[1] for elem in self.generate_weights_and_biases((data.shape[1], hidden_dim), num_generations)]

        # Set activation function and store name for serialization
        self.activation_name = activation_name
        self.activation_function = {'linear': self.linear,
                                    'sigmoid': self.sigmoid,
                                    'tanh': self.tanh,
                                    'sin': self.sin,
                                    'relu': self.relu}[activation_name]

        # Compute node outputs: concatenate all projections
        nodes = data.dot(self.weight_list[0]) + self.bias_list[0]
        for i in range(1, len(self.weight_list)):
            nodes = np.column_stack((nodes, data.dot(self.weight_list[i]) + self.bias_list[i]))

        return self.activation_function(nodes)

    def compute_node_inputs(self, test_data):
        """
        Compute pre-activation values (before applying activation function).

        This is needed for derivative calculations in the solver.

        Args:
            test_data: Input data array

        Returns:
            Pre-activation values Z = X @ W + b
        """
        transformed_nodes = test_data.dot(self.weight_list[0]) + self.bias_list[0]
        for i in range(1, len(self.weight_list)):
            transformed_nodes = np.column_stack((transformed_nodes, test_data.dot(self.weight_list[i]) + self.bias_list[i]))
        return transformed_nodes

    def transform(self, test_data):
        """
        Transform test data through the node layer (forward pass).

        Args:
            test_data: Input data array

        Returns:
            Activated node outputs
        """
        output = self.compute_node_inputs(test_data)
        return self.activation_function(output)

    def get_feature_var_weights(self, var_index):
        """
        Extract weights for a specific input variable across all node groups.

        Used for computing derivatives with respect to a specific variable.

        Args:
            var_index: Index of input variable (e.g., 0 for x, 1 for t)

        Returns:
            Concatenated weights for the specified variable
            Shape: (1, num_generations * hidden_dim)
        """
        weight = self.weight_list[0][var_index:var_index+1, :]
        for i in range(1, len(self.weight_list)):
            weight = np.column_stack((weight, self.weight_list[i][var_index:var_index+1, :]))
        return weight

    def get_enhance_weights(self):
        """
        Get concatenated weight matrix for all enhancement nodes.

        Used for computing derivatives through the enhancement layer.

        Returns:
            Full weight matrix of shape (input_dim, total_hidden_dim)
        """
        weight = self.weight_list[0]
        for i in range(1, len(self.weight_list)):
            weight = np.column_stack((weight, self.weight_list[i]))
        return weight

    def state_dict(self):
        """
        Extract generator state for saving.

        Returns:
            Dictionary containing all state needed to restore the generator:
            - weight_list: List of weight matrices (numpy arrays)
            - bias_list: List of bias scalars
            - activation_name: Name of activation function
            - whiten: Whether orthogonalization is enabled
            - random_range: Range for weight initialization
        """
        return {
            'weight_list': [w.copy() for w in self.weight_list],
            'bias_list': list(self.bias_list),
            'activation_name': self.activation_name,
            'whiten': self.whiten,
            'random_range': self.random_range,
        }

    def load_state_dict(self, state_dict):
        """
        Restore generator state from saved state.

        Args:
            state_dict: Dictionary from state_dict() method
        """
        self.weight_list = [w.copy() for w in state_dict['weight_list']]
        self.bias_list = list(state_dict['bias_list'])
        self.activation_name = state_dict['activation_name']
        self.whiten = state_dict['whiten']
        self.random_range = state_dict['random_range']

        # Restore activation function from name
        if self.activation_name is not None:
            self.activation_function = {
                'linear': self.linear,
                'sigmoid': self.sigmoid,
                'tanh': self.tanh,
                'sin': self.sin,
                'relu': self.relu
            }[self.activation_name]
