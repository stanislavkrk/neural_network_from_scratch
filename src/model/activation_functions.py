"""
Module: activation_functions

This module provides common activation functions used in neural networks,
including Sigmoid and Softmax variants. These functions add nonlinearity
to the model, allowing it to learn more complex representations of data.
"""

import numpy as np

def sigmoid(array_input):
    """
    Sigmoid activation function. It normalizes the input data to the range (0,1),
    providing non-linearity in neural network training, allowing the model to learn
    complex dependencies.

    :param array_input: z1 logits
    :return: The array after applying the sigmoid activation.
    """
    sigmoid = (1 / (1 + np.exp(-array_input)))
    return sigmoid

def softmax(data):
    """
    Softmax activation function. It normalizes the input data to the range (0,1),
    providing non-linearity in neural network training, allowing the model to learn
    complex dependencies.

    :param data: z2 logits
    :return: The array after applying the softmax activation (a2).
    """
    data = np.array(data).reshape(11, -1)
    exp_data = np.exp(data - np.max(data, axis=1, keepdims=True))
    softmax_data = exp_data / np.sum(exp_data, axis=1, keepdims=True)
    return softmax_data

def softmax_mono(data):
    """
    Softmax calculation function for normalizing input data.
    Works with a 1D array (representing a single letter).

    :param data: z2 logits for a single letter
    :return: The array after applying softmax for that single letter (a2).
    """
    data = np.array(data).flatten()  # Convert data to a one-dimensional array
    exp_data = np.exp(data - np.max(data))  # Subtract the maximum for numerical stability
    return exp_data / np.sum(exp_data)
