import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    if rng is None:
        rng = np.random.default_rng()
    
    if rng.random()<epsilon:
        return rng.integers(0,len(q_values))
    return np.array(q_values).argmax()