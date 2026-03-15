import numpy as np

def create_model(size, n_embedding):
    model = {
        "W1" : np.random.randn(size,n_embedding),
        "W2" : np.random.rand(n_embedding,size)
    }
    return model

def softmax(matrix):
    matrix = matrix - np.max(matrix,axis=1,keepdims=True)
    exponential = np.exp(matrix)
    return exponential / np.sum(exponential, axis=1, keepdims=True)

def cross_entropy(z,y):
    return - np.sum(np.log(z+1e-9)*y)

def forward(model,matrix):
    cache = {}

    cache["A1"] = matrix @ model["W1"]
    cache["A2"] = cache["A1"] @ model["W2"]
    cache["Z"] = softmax(cache["A2"])

    return cache

def backward(model,matrix,y,alpha):
    cache = forward(model,matrix)
    da2 = cache["Z"] - y
    dw2 = cache["A1"].T @ da2
    da1 = da2 @ model["W2"].T
    dw1 = matrix.T @ da1
    assert(dw2.shape == model["W2"].shape)
    assert(dw1.shape == model["W1"].shape)
    model["W1"] -= alpha * dw1
    model["W2"] -= alpha * dw2
    return cross_entropy(cache["Z"],y)
