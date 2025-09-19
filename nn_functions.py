# This is the file where all the steps of generating a Neural Network are 
# generated in form of functions, for the consecutive deployment and reuse
#
# All the functions will be done from scratch with NumPy framework, trying
# to ensure Vectorization for optimized performance.

import numpy as np
import copy

# 1. Initialization of parameters (Weights - W and bias - b)
#    Weights will be initialized to random values in order to break simmety
#    and allow the neurons to learn different features of the data.

def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):

        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
        # check that weight and bias shape are correct
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# 2. Implement the linear part of a layer's forward propagation
#    Z[l] = W[l]xA[l-1] + b[l]

def linear_forward(A, W, b):
    """
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b   
    cache = (A, W, b)
    
    return Z, cache


# 3. Implement the activations functions

# 3.1 Sigmoid activation function - g(Z) = 1 / (1+e^-Z) 

def sigmoid(Z):
    """
    Arguments:
    Z -- the input of the activation function, also called pre-activation parameter

    Returns:
    A -- activation value
    cache -- a variable that contains "Z"; stored for computing the backward pass efficiently
    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

# 3.2 ReLU activation function - g(Z) = max(0,Z)

def relu(Z):
    """
    Arguments:
    Z -- the input of the activation function, also called pre-activation parameter

    Returns:
    A -- activation value
    cache -- a variable that contains "Z"; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)
    cache = Z

    return A, cache


# 4. Implement the forward propagation for the "Linear Activation" layer
#    Z[l] = W[l]xA[l-1] + b[l]
#    A[l] = g[l](Z[l]

def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
        
    cache = (linear_cache, activation_cache)

    return A, cache


# 6. Implement the full forward propagation - in this Neural Net the hidden layers will activate with 
#    the "ReLU" function, and the final Lth layer with "Sigmoid", as it will be used for binary classification

def forward_prop(X, parameters):
    """
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2     # number of layers in the neural network (params is composed of layer_dims * each pair of W/b)
    
    # Implement (L-1) layers: Linear -> ReLU
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation="relu")
        caches.append(cache)
        
    
    # Implement the L layer: Linear -> Sigmoid
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation="sigmoid")
    caches.append(cache)
    # YOUR CODE ENDS HERE
          
    return AL, caches


# 7. Implementation of the binary cross entropy cost function (log loss) - for binary classification the Logistic Regression Cost Function is used:
#    J(W,b) =  -1/m  * summatory(Y * log(A) + (1 - Y) * log(1 - A))

def cost_function(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to label predictions, shape (1, number of examples)
    Y -- true "label" vector (0, 1 - binary variable)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = (-1/m)*np.sum((Y*np.log(AL))+(1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)      # To make sure the cost's shape is what is expect (e.g. this turns [[17]] into 17).

    
    return cost


# 8. Implementation of the linear portion of backpropagation for a single layer

def linear_backward(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# 9. Implement the backpropagation of the activation functions

# 9.1 Sigmoid - g'(Z) = 1 /(1 + e^-Z) * (1 - 1 /(1 + e^-Z))

def  sigmoid_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single sigmoid unit.

    Arguments:
    dA -- post-activation gradient (numpy array, same shape as activation_cache)
    activation_cache -- 'Z' stored during forward propagation

    Returns:
    dZ -- gradient of the cost with respect to Z
    """
    Z = activation_cache
    A = 1 / (1 + np.exp(-Z))          # Recompute sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ

# 9.2 ReLU - g'(Z) = 0 if Z < 0, 1 if Z >= 0

def relu_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single ReLU unit.

    Arguments:
    dA -- post-activation gradient (numpy array, same shape as activation_cache)
    activation_cache -- 'Z' stored during forward propagation

    Returns:
    dZ -- gradient of the cost with respect to Z
    """
    Z = activation_cache
    dZ = np.array(dA, copy=True) # converting dA to a correct object.

    # When Z <= 0, set dZ to 0 as well.
    dZ[Z <= 0] = 0
    return dZ


# 10. Implement the full backpropagation of the linear activation layer.

def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)       
    
    return dA_prev, dW, db


# 11. Implement the full backpropagation : 
#     backpropagation of cost -> Lth layer - Sigmoid -> Linear -> L-1 layer  - ReLU -> Linear 

def backward_prop(AL, Y, caches):
    """
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
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
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# 12. Implement gradient descent with the update of Weights an biases

def update_parameters(params, grads, learning_rate):
    """
    Arguments:
    params -- python dictionary containing parameters 
    grads -- python dictionary containing gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = params['W' + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = params['b' + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters