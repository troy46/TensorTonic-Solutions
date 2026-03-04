import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.array(w,dtype='float64')
    g = np.array(g,dtype='float64')
    s = np.array(s,dtype='float64')
    # Write code here
    s = beta*s + (1-beta)*g*g
    w = w - lr/np.sqrt(s+eps)*g
    return w, s