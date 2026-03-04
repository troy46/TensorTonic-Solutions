import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    # Write code here
    Q = np.array(Q,dtype='float64')
    td_tar = r + gamma*Q.max(axis=1)[s_next]
    Q[s, a] = Q[s, a] + alpha*(td_tar-Q[s, a])
    return Q.tolist()