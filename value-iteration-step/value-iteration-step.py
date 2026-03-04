import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    v = (np.array(transitions)*np.array(values,dtype='float64')\
            .reshape((1,1,-1))).sum(axis=2)
    q = np.array(rewards,dtype='float64')+v*gamma
    return q.max(axis=1).tolist()