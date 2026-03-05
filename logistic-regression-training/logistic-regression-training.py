import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _d_sigmoid(z):
    return np.where(z >= 0, np.exp(-z)/(1+np.exp(-z))**2, np.exp(z)/(1+np.exp(z))**2)

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    
    
    def gradient(w,b):
        p = _sigmoid(X @ w + b)
        dl_dp = -(y/p - (1-y)/(1-p))/len(y) #(N,1)
        dp_dw = X.T*_d_sigmoid(X@w+b) #(p,N)
        dp_db = _d_sigmoid(X@w+b) # (N,1)
        return dp_dw@dl_dp, np.inner(dp_db,dl_dp)

    w, b = np.zeros(X.shape[1]), 0
    for _ in range(steps):
        dw, db = gradient(w, b)
        w -= lr*dw
        b -= lr*db
    return w, b