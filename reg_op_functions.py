# This is the file where regularization and optimization functions will 
# be generated, for the consecutive deployment and reuse

import numpy as np
import nn_functions

# 1. Initialization of parameters (Weights - W and Biases - B)

# 1.1. Random Initialization 2.0
#   It is basically the same function as the initalization of parameters in 'nn_functions.py',
#   but with a modification that allows the user choose the scale of the random intialization
#   of the weights.

def initialize_parameters_random(layers_dims, scale):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    scale -- float number to control the scaling of the random initialization of parameters

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(42)    # This is just to obtain same parameters - feel free of turning it off
    parameters = {}
    L = len(layers_dims)            
    
    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * scale
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

# 1.2.'He' Initialization of parameters
#   An alternative method of initializing the weights from the random initialization.
#   It multiplies the random initialization by (2 / dim. of previous layer)^1/2
#   'He' Initialization improves the performance of layers with ReLU activation

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(42)
    parameters = {}
    L = len(layers_dims) - 1 
     
    for l in range(1, L + 1):

        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters


# 2. Regularization - This methods are implemented to reduce overfitting in the NNets,
#    improving the performance in the test set and helping the system overall.

