import numpy as np
def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n = len(y_pred)
    bin_assignment = np.floor(y_pred*n_bins).clip(0,n_bins-1)
    ib = np.identity(n_bins)
    bin_mat = np.asarray([ib[int(b),:] for b in bin_assignment])
    bin_count = bin_mat.sum(axis=0)
    bin_pred = (y_pred.reshape(n,1)*bin_mat).sum(axis=0)
    bin_true = (y_true.reshape(n,1)*bin_mat).sum(axis=0)
    
    return abs(bin_pred-bin_true).sum()/n
    