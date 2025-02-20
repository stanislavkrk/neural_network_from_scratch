"""
Module: new_NN

This module provides the core components of a simple fully-connected neural network,
implemented from scratch. It includes:

- Definitions for the input/hidden layer (NN_input_and_hidden_layer) and the output layer (NN_output_layer).
- Initialization of random weights and biases.
- The cross-entropy loss function.
- Backpropagation and weight update procedures.
- A training loop (training) that ties everything together.
- A prediction function (prediction) to classify new inputs.

Overall, it showcases how to build a basic feedforward neural network from first principles:
forward pass, loss computation, backpropagation, gradient descent updates, and final inference.
"""

import numpy as np
from .activation_functions import sigmoid, softmax, softmax_mono

def NN_input_and_hidden_layer(list_input, w1, b1):
    """
    The input layer of the neural network and the first hidden layer.
    The input layer directly receives the data encoding the letters, where each letter is
    represented by a vector of 30 elements (0s and 1s).
    There are 30 input neurons — one for each element in the input vector. However, these neurons
    essentially just pass the signals through.

    In this function, we perform a matrix multiplication (dot) between the input vector (x)
    and the weight matrix (w), then add the bias (b).

    The input vector has the same number of elements as the total features used to encode each letter.
    The weight matrix between the input layer and the hidden layer has the shape
    (# of input neurons) × (# of hidden neurons).
    We set the number of hidden neurons at our discretion. The hidden neurons receive all input features
    and perform the necessary computations.

    :param list_input: An array of input data, each a feature vector for a letter (binary code).
    :param w1: The weight matrix.
    :param b1: The bias matrix.
    :return: z1, the logits (raw output values that are the linear combination of weights and inputs,
             plus the bias) for the hidden layer.
    """
    x = np.array(list_input)

    # Matrix multiplication of each letter’s vector with the weight matrix + adding bias
    z1 = np.dot(x, w1) + b1  # Multiply and add the bias

    return z1

def NN_output_layer(data, w2, b2):
    """
    The output layer of the neural network. In the output layer, the number of neurons (units)
    depends on how many classes we need to classify. The number of input neurons here is the same
    as the number of hidden neurons in the previous layer (they processed the data).

    We do a similar matrix multiplication (dot) for the specified number of output neurons.

    :param data: The logits z1 after sigmoid activation, i.e., a1.
    :param w2: The weight matrix for the output layer.
    :param b2: The bias for the output layer.
    :return: z2, the logits of the output layer.
    """
    x = data
    z2 = []
    for x_i in x:
        z2_i = x_i.dot(w2) + b2
        z2.append(z2_i)
    z2 = np.array(z2)

    return z2

def random_weights_and_bias(x=30,y=15):
    """
    A generator for random weights and biases to initialize training.

    :param x: The size of the feature vector for each letter (default 30).
    :param y: The number of hidden neurons (default 15).
    :return: Randomly initialized weights and biases.
    """
    # Create the initial random weight matrix (30 input neurons × 15 hidden neurons)
    w1 = np.random.rand(x,y) * 0.01

    # Create the bias vector (15 neurons in the hidden layer)
    b1 = np.random.uniform(-0.1, 0.1, (1,y))
    return w1, b1

def cross_entropy_loss(y_true, y_pred):
    """
    Compute the loss function using the average cross-entropy with the target confidence
    (true labels) and the predicted probabilities. Cross-entropy works well for classification tasks.

    The total loss is divided by the number of examples (y_true.shape[0]).
    * shape[0] refers to the number of rows in the NumPy array (the number of examples).

    :param y_true: The true labels, represented as one-hot encodings.
    :param y_pred: The predicted probability for each class, returned by the Softmax function.
    :return: The mean cross-entropy loss.
    """
    loss = - np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
    return loss

def back_propagation(x, y_true, a1, a2, w2, b2, w1, b1):
    """
    Compute the error vectors (dZ) in both the output and hidden layers. These indicate how much
    the model’s output differs from the target value. For a loss function, the error vector
    is the gradient of this function over the predicted values (y_pred).
    It shows in which direction and by how much y_pred should be changed to reduce the error (loss).

    :param x: The input data (letter feature vectors).
    :param y_true: The true labels, represented as one-hot encodings.
    :param a1: The hidden layer activations after the sigmoid.
    :param a2: The output layer activations after Softmax.
    :param w2: The weights of the output layer.
    :param b2: The biases of the output layer.
    :param w1: The weights of the hidden layer.
    :param b1: The biases of the hidden layer.
    :return: All gradients required to update the network’s weights.
    """
    ## Output layer

    # 1. The error at the output layer  is the derivative of the loss function (cross-entropy)
    # in relation to the logits of z2.
    # With the simplification for Softmax + CrossEntropy, the output error is equal to this difference:
    #  a2 (predicted probabilities) - y_true (true values).
    dL_dz2 = a2 - y_true

    # 2. Gradient for the output layer’s weight coefficients w2. This is the derivative of the logit z2 by the weights w2.
    # This is based on the fact that weights affect z2 through a1 (the hidden layer output).
    dL_dw2 = np.dot(a1.T, dL_dz2)

    # 3. Gradient for the bias b2 — it’s the sum of the errors for each class (logit) in the output layer,
    # since bias b2 is added to each logit independently.
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)


    ## Hidden layer

    # 1. This is the hidden layer error. We backpropagate the output error to the hidden layer
    # through the transposed weights w2.
    dL_da1 = np.dot(dL_dz2, w2.T)

    # 1.2 Multiplication by the sigmoid derivative. This allows to calculate the error
    # at the logit level z1 of the hidden layer.
    dL_dz1 = dL_da1 * a1 * (1-a1)

    # 2. Gradient for weighting coefficients w1. This is the derivative of the logits z1 of the hidden layer by the weights w1.
    # The formula is based on the fact that the weights w1 affect the logits z1 through the input x.
    dL_dw1 = np.dot(x.T, dL_dz1)

    # 3. Gradient for bias b1. This is the sum of the errors for each neuron of the hidden layer, since the bias b1
    # is added to each of the logits of z1 independently.
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    return dL_dw1, dL_dw2, dL_db1, dL_db2

def update_weights(w1, b1, w2, b2, dL_dw1, dL_db1, dL_dw2, dL_db2, learning_rate):
    """
    Update the weights in the training algorithm.

    :param w1: Weights for the hidden layer.
    :param b1: Biases for the hidden layer.
    :param w2: Weights for the output layer.
    :param b2: Biases for the output layer.
    :param dL_dw1: Gradient of the loss w.r.t. hidden layer weights w1.
    :param dL_db1: Gradient of the loss w.r.t. hidden layer bias b1.
    :param dL_dw2: Gradient of the loss w.r.t. output layer weights w2.
    :param dL_db2: Gradient of the loss w.r.t. output layer bias b2.
    :param learning_rate: The learning rate.
    :return: Updated weight matrices and biases.
    """

    # Update weights for the hidden layer
    w1 -= learning_rate * dL_dw1
    b1 -= learning_rate * dL_db1

    # Update weights for the output layer
    w2 -= learning_rate * dL_dw2
    b2 -= learning_rate * dL_db2

    return w1, b1, w2, b2

def training(x, w1, b1, w2, b2, y_true):
    """
    Training loop using gradient descent.

    :param x: Input data, feature vectors.
    :param w1: Hidden layer weights.
    :param b1: Hidden layer biases.
    :param w2: Output layer weights.
    :param b2: Output layer biases.
    :param y_true: Target confidence (true labels).
    :return: The trained model weights.
    """
    # Call the layers and activation functions, then train the neural network
    for epoch in range(1000):

        z1 = NN_input_and_hidden_layer(x, w1, b1)
        a1 = sigmoid(z1)
        z2 = NN_output_layer(a1, w2, b2)
        a2 = softmax(z2)

        loss = cross_entropy_loss(y_true, a2)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

        dL_dw1, dL_dw2, dL_db1, dL_db2 = back_propagation(x, y_true, a1, a2, w2, b2, w1, b1)

        learning_rate = 0.1
        w1, b1, w2, b2  = update_weights(w1, b1, w2, b2, dL_dw1, dL_db1, dL_dw2, dL_db2, learning_rate)

    return w1, b1, w2, b2

def prediction (x, w1, b1, w2, b2, letters_list):
    """
    Class prediction function. In this case, it predicts the letter class.
    It uses a variant of the Softmax activation for a single input letter.
    The recognized class is chosen via np.argmax on the array of probabilities (a2).

    :param x: The feature vector of the letter to be recognized.
    :param w1: Weights of the trained model's hidden layer.
    :param b1: Biases of the trained model's hidden layer.
    :param w2: Weights of the trained model's output layer.
    :param b2: Biases of the trained model's output layer.
    :param letters_list: A list of letter names.
    :return: The index of the recognized class.
    """
    z1 = NN_input_and_hidden_layer(x, w1, b1)
    a1 = sigmoid(z1)
    z2 = NN_output_layer(a1, w2, b2)
    a2 = softmax_mono(z2)

    # Identify the class with the highest probability
    predicted_class = np.argmax(a2)
    print(f"Ймовірності класів: {a2}")
    print(f"Прогнозований клас: {predicted_class}")
    print('Літера', letters_list[predicted_class])
    return predicted_class