def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    topk = recommended[:k]
    overlap = set(topk) & set(relevant)
    precision_k = len(overlap)/len(topk)
    recall_k = len(overlap)/len(relevant)
    return [precision_k, recall_k]