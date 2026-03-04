import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # Write code here
    running_min = np.ones((1,D))*float('inf')
    running_max = np.ones((1,D))*float('-inf')
    return {'running_min':running_min, 'running_max':running_max}

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Write code here
    X_batch = np.asarray(X_batch)
    state['running_min'] = np.minimum(state['running_min'], X_batch.min(axis=0,keepdims=True))
    state['running_max'] = np.maximum(state['running_max'], X_batch.max(axis=0,keepdims=True))

    return (X_batch - state['running_min'])/(state['running_max']-state['running_min']+eps)
    