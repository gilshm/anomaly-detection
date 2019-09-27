import torch
import Config as cfg
import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
from sklearn import metrics


def auroc(score, labels, target, samples=1e5):
    TPR = []
    FPR = []

    normal_idx = labels == target           # Positive
    anomaly_idx = labels != target          # Negative
    condition_positive = normal_idx.sum()   # TP + FN
    condition_negative = anomaly_idx.sum()  # TN + FP

    for eta in np.linspace(0, 1, samples):
        positive_class = score > eta

        true_pos = positive_class[normal_idx].sum()     #TP - total number of true positive results
        false_pos = positive_class[anomaly_idx].sum()   #FP - total number of false positive results

        TPR.append((true_pos.float()/condition_positive).cpu().numpy().item())
        FPR.append((false_pos.float()/condition_negative).cpu().numpy().item())

    roc_auc = metrics.roc_auc_score(normal_idx.cpu().numpy(), score.cpu().numpy())
    print("AUC-ROC = {}".format(roc_auc))

    return roc_auc
