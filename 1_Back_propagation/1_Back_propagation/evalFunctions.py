import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """


    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------

    acc = np.mean(LPred == LTrue)
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------

    labels = set(LPred)

    cM = np.zeros((len(labels), len(labels)))

    for i, pred_lab in enumerate(labels):
        for j, actual_lab in enumerate(labels):
            cM[i, j] = np.sum((LPred == pred_lab) & (LTrue == actual_lab))

    # ============================================

    cM = cM.astype(int)
    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc = None
    # ============================================
    
    return acc
