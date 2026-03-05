import numpy as np
def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    similarities = np.asarray(similarities).clip(0,None)
    ratings = np.asarray(ratings)
    if similarities.sum() == 0:
        return 0
    return (similarities*ratings).sum()/similarities.sum()