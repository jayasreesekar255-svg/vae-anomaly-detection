from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score

def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Given true binary labels and anomaly scores (higher => more anomalous),
    compute AUC-ROC and AUPRC (average precision), and a simple precision@k.
    """
    assert y_true.shape[0] == scores.shape[0]
    # invert scores if necessary: here higher score == more anomalous (reconstruction error)
    aucroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    # precision@top_k where k = number of true anomalies
    k = max(1, int(y_true.sum()))
    idx_sorted = np.argsort(-scores)
    topk = idx_sorted[:k]
    precision_at_k = y_true[topk].sum() / k

    return {"AUC_ROC": float(aucroc), "AUPR": float(aupr), "P@k": float(precision_at_k)}
