import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    fpr.sort()
    tpr.sort()

    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr length mismatch")
    auc = 0
    for i in range(len(fpr)-1):
        auc += (fpr[i+1]-fpr[i])*(tpr[i]+tpr[i+1])/2
    return auc