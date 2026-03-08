import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x)
    return (1-np.exp(-2*abs(x)))/(1+np.exp(-2*abs(x)))*((x>0)*2-1)
