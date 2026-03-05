import numpy as np
def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    series = np.asarray(series)
    y = series-series.mean()
    total_var = (y**2).sum()
    if total_var == 0:
        return [1]+[0]*max_lag
    auto_corr = []
    for lag in range(max_lag+1):
        auto_corr.append((y[lag:]*y[:(len(y)-lag)]).sum()/total_var)
    return auto_corr