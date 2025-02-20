"""
Module: data_input_mod

This module provides functions for handling and transforming letter data, including:
- Creating a list of letter names from a given dictionary (letters_list).
- Converting dictionary values into NumPy arrays of encoded features (letters_data).
- Generating one-hot encodings for a set of letters (generate_one_hot_encoding).
"""

import numpy as np

def letters_list(letters_dict):
    """
    This function takes a dictionary provided by the data_input function.
    The dictionary has letters as keys and their binary (visual) encodings as values.

    :param letters_dict: A dictionary containing letter encodings.
    :return: A list of letter names.
    """
    dict = letters_dict
    letters_list = [name for name in dict.keys()]
    return letters_list

def letters_data(letters_dict):
    """
    This function takes a dictionary provided by the data_input function.
    The dictionary has letters as keys and their binary (visual) encodings as values.

    :param letters_dict: A dictionary containing letter encodings.
    :return: A list of NumPy arrays, each representing the encoding values.
    """
    dict = letters_dict
    letters_data = [value for value in dict.values()]
    letters_data = np.array(letters_data)
    # Reshape to ensure each encoding is treated as a row
    letters_data.reshape(letters_data.shape[0], -1)
    return letters_data

def generate_one_hot_encoding(letters_list):
    """
    Automatically creates one-hot encoding for any number of letters.
    This is the output (label) part of the neural network training set.
    These data serve as labels for the corresponding input data.

    One-hot encoding is a way of representing categorical data as a binary vector, where:
      - One element is set to 1 (corresponding to the current class),
      - The rest of the elements are 0.

    np.eye(n) generates an identity matrix of size n×n, where each row is a binary representation
    for the corresponding letter.

    :param letters_list: A list of letter names.
    :return: A NumPy array containing one-hot vectors for each letter.
    """
    n = len(letters_list)
    labels = np.eye(n)
    return np.array(labels)
