import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # YOUR CODE HERE
    x = x.T
    for gradient in gradients_F:
        x = (gradient+np.identity(gradient.shape[0])) @ x
    return x.T


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    x = x.T
    for gradient in gradients_F:
        x = gradient @ x
    return x.T
