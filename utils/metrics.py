"""
utils/metrics.py
Responsabilité : calculer les métriques d'évaluation d'un modèle IA.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import numpy as np


def compute_metrics(y_true, y_pred):
    """
    Calcule les métriques classiques de classification.
    
    :param y_true: vraies étiquettes
    :param y_pred: prédictions du modèle
    :return: dictionnaire contenant toutes les métriques
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics


def compute_roc_auc(y_true, y_proba):
    """
    Calcule l'aire sous la courbe ROC (AUC).
    Nécessite les probabilités, pas juste les classes.
    
    :param y_true: vraies étiquettes
    :param y_proba: probabilités de la classe positive
    :return: score AUC
    """
    try:
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return None  # Cas où une seule classe est présente


def print_metrics_report(metrics, auc=None):
    """
    Affiche joliment les métriques dans la console.
    """
    print("\n" + "=" * 50)
    print("📊 RAPPORT D'ÉVALUATION DU MODÈLE")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    if auc is not None:
        print(f"  AUC-ROC   : {auc:.4f}")
    print("-" * 50)
    print("  Matrice de confusion :")
    cm = np.array(metrics["confusion_matrix"])
    print(f"    TN={cm[0][0]:>3}  FP={cm[0][1]:>3}")
    print(f"    FN={cm[1][0]:>3}  TP={cm[1][1]:>3}")
    print("=" * 50 + "\n")


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Affiche le rapport complet de scikit-learn.
    """
    if target_names is None:
        target_names = ["Sain", "Infecté"]
    print("\n📋 Rapport détaillé :")
    print(classification_report(y_true, y_pred, target_names=target_names))