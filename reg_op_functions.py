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

# 2.1. L2 Regularization
#      This regularization method updates the cost function with a weight decay. Weights 
#      are pushed to smaller values, leading to a smoother model and reducing overfitting.
#      The implementation is done by modifying the cost function adding:
#      + (lambda / 2 *m) * Frobenius norm of the Weights. (Frobenius norm is the sum of squared elements of a matrix)

def compute_cost_with_l2_reg(A, Y, parameters, lambd, layers_dims):
    """
    Implement the cost function with L2 regularization.
    
    Arguments:
    A -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    lambd -- regularization parameter
    layer_dims -- python array (list) containing the size of each layer

    Returns:
    cost - value of the regularized loss function
    """
    m = Y.shape[1]
    L = len(layers_dims) - 1
    
    cross_entropy_cost = cost_function(A, Y) # This gives you the cross-entropy part of the cost
    L2_regularization_cost = (1/m)*(lambd/2) * sum(np.sum(np.square(parameters['W' + str(l)])) for l in range(1, L + 1))
    # inner np.sum(np.square(Wl)) computes the sum of all squared elements
    # sum adds up these scalars for all layers
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

# In brackpropagation the regularization term must be added to the derivatives of the weights:
# d/dW((lambda/2*m) * W^2) = (lambda/m) * W

def backward_prop_with_l2_reg(AL, Y, caches, lambd, parameters):
    """
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (0, 1 - binary variable)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    lambd -- regularization parameter
    parameters -- python dictionary containing parameters of the model

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # make Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # grad of the cost function
    

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    dW_temp = dW_temp + (lambd / m) * parameters['W' + str(L)]
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation='relu') + (lambd / m) * parameters['W' + str(l)]
        dW_temp = dW_temp + (lambd / m) * parameters['W' + str(l + 1)]
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# 2.2 Dropout Regularization:
#     This regularization method consists in shutting down different neurons in each layer,
#     so after each iteration a different model is trained using only a subset of neurons.
#     This makes the neurons become less sensitive to the activation of one other specific 
#     neuron, making the model overall smoother.

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- python dictionary containing parameters of the model
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """ 
    np.random.seed(42) # for reproductibility

    caches = []
    A = X
    L = len(parameters) // 2     # number of layers in the neural network (params is composed of layer_dims * each pair of W/b)
    
    # Implement (L-1) layers: Linear -> ReLU
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation="relu")
        D = np.random.rand(A.shape[0], A.shape[1])      # Step 1: initialize matrix D = np.random.rand(..., ...)
        D = (D < keep_prob).astype(int)                 # Step 2: convert entries of D to 0 or 1 (using keep_prob as the threshold)
        A = A*D                                         # Step 3: shut down some neurons of A
        A = A/keep_prob                                 # Step 4: scale the value of neurons that haven't been shut down
        caches.append((cache, D))                       # Mask 'D' is stored for backpropagation
        
    
    # Implement the L layer: Linear -> Sigmoid
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation="sigmoid")
    caches.append((cache, None))
    
          
    return AL, caches


def backward_prop(AL, Y, caches, keep_prob):
    """
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (0, 1 - binary variable)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    grads -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables 
    """
    grads = {}
    L = len(caches)  # number of layers
    Y = Y.reshape(AL.shape)
    
    # Output layer (sigmoid): no dropout
    cache, D = caches[L-1]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, cache, activation='sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Hidden layers (ReLU + dropout)
    for l in reversed(range(L-1)):
        cache, D = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], cache, activation='relu')
        dA_prev_temp = dA_prev_temp * D                     # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
        dA_prev_temp = dA_prev_temp / keep_prob             # Step 2: Scale the value of neurons that haven't been shut down
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads