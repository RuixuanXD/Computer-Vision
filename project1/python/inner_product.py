import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape
    n = param["w"].shape[1]

    ###### Fill in the code here ######
    x = input["data"]
    b = param["b"].T
    w = param["w"].T
    y = np.dot(w, x) + b
    h = y.shape[1]
    #print(y[1])
    
    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": y # replace 'data' value with your implementation
    }

    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    dl_dh = output['diff']
    #print(dl_dh)
    X = input_data['data']
    w = np.dot(X, dl_dh.T)
    b = np.sum(dl_dh.T, axis=0)
    l = np.dot(param['w'], dl_dh)
    param_grad['b'] = b
    param_grad['w'] = w
    input_od = l

    return param_grad, input_od