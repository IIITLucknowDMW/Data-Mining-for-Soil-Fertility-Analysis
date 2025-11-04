import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def distance(instance1, instance2, method):
    """
    Calculate distance between two instances
    
    Parameters:
    - method: 0 for Cosine, any other number for Minkowski with that order
    """
    if method == 0:  # Cosine
        return 1 - ((np.sum([instance1[i] * instance2[i] for i in range(0, len(instance1))])) / 
                    (np.sqrt(np.sum([i**2 for i in instance1])) * np.sqrt(np.sum([i**2 for i in instance2]))))
    else:  # Minkowski
        return sum(np.abs(instance1 - instance2)**method)**(1 / method)


def confusion_matrix_custom(y_test, y_pred):
    N = len(np.unique(y_test))
    M = np.zeros((N, N), dtype=int)
    for i in range(0, y_test.shape[0]):
        M[int(y_test[i])][int(y_pred[i])] += 1
    return M


def values(m):
    TP = m.diagonal()
    FP = m.sum(axis=0) - TP
    FN = m.sum(axis=1) - TP
    TN = m.sum() - (TP + FN + FP)
    return TP, FN, FP, TN


def recall_score_custom(TP, FN):
    return TP / (TP + FN)


def precision_score_custom(TP, FP):
    return TP / (TP + FP)


def false_positive_rate(FP, TN):
    return FP / (FP + TN)


def specificity_score_custom(TN, FP):
    return TN / (TN + FP)


def accuracy_score_custom(M):
    return np.sum(M.diagonal()) / np.sum(M)


def f1_score_custom(TP, FP, FN):
    recall = recall_score_custom(TP, FN)
    precision = precision_score_custom(TP, FP)

    if np.any(np.isnan(recall)) or np.any(np.isnan(precision)) or np.all(recall + precision == 0):
        f1 = 0.0
    else:
        f1 = 2 * (recall * precision) / (recall + precision)

    return f1


def classification_metrics(y_test, y_pred):
    """
    Calculate and display classification metrics
    
    Parameters:
    - y_test: true labels
    - y_pred: predicted labels
    """
    M = confusion_matrix_custom(y_test, y_pred)
    TP, FN, FP, TN = values(M)
    
    print(f"Global Accuracy: {np.mean(accuracy_score_custom(M)):.4f}")
    print(f"Global Recall: {np.mean(recall_score_custom(TP, FN)):.4f}")
    print(f"Global Precision: {np.mean(precision_score_custom(TP, FP)):.4f}")
    print(f"Global Specificity: {np.mean(specificity_score_custom(TN, FP)):.4f}")
    print(f"Global F1 Score: {np.mean(f1_score_custom(TP, FP, FN)):.4f}")

    print(f"\nPer-class Recall: {recall_score_custom(TP, FN)}")
    print(f"Per-class Precision: {precision_score_custom(TP, FP)}")
    print(f"Per-class False Positive Rate: {false_positive_rate(FP, TN)}")
    print(f"Per-class Specificity: {specificity_score_custom(TN, FP)}")
    print(f"Per-class F1 Score: {f1_score_custom(TP, FP, FN)}")
    
    sns.heatmap(M, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
