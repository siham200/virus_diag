"""
pipeline/evaluator.py
Responsabilité : évaluer les performances d'un modèle IA entraîné.
"""

import numpy as np
from utils.metrics import (
    compute_metrics,
    compute_roc_auc,
    print_metrics_report,
    print_classification_report
)


class Evaluator:
    """
    Classe qui évalue un modèle IA sur des données de test.
    """

    def __init__(self, model, target_names=None):
        """
        :param model: modèle IA déjà entraîné (avec predict() et predict_proba())
        :param target_names: noms des classes (ex: ["Sain", "Infecté"])
        """
        if model is None:
            raise ValueError("Le modèle ne peut pas être None.")
        
        self.model = model
        self.target_names = target_names or ["Sain", "Infecté"]
        
        # Résultats de l'évaluation
        self.y_pred = None
        self.y_proba = None
        self.metrics = None
        self.auc = None

    def evaluate(self, X_test, y_test, verbose=True):
        """
        Évalue le modèle sur les données de test.
        
        :param X_test: features de test
        :param y_test: vraies étiquettes de test
        :param verbose: si True, affiche le rapport dans la console
        :return: dictionnaire des métriques
        """
        print(f"\n🔍 Évaluation du modèle {self.model.__class__.__name__}...")
        
        # 1. Prédictions
        self.y_pred = self.model.predict(X_test)
        
        # 2. Probabilités (si disponibles)
        try:
            proba_all = self.model.predict_proba(X_test)
            self.y_proba = proba_all[:, 1]  # Probabilité de la classe positive
        except AttributeError:
            self.y_proba = None
            print("   ⚠️  Le modèle ne supporte pas predict_proba(). AUC non calculé.")
        
        # 3. Calcul des métriques
        self.metrics = compute_metrics(y_test, self.y_pred)
        
        # 4. Calcul de l'AUC si possible
        if self.y_proba is not None:
            self.auc = compute_roc_auc(y_test, self.y_proba)
        
        # 5. Affichage
        if verbose:
            print_metrics_report(self.metrics, self.auc)
            print_classification_report(y_test, self.y_pred, self.target_names)
        
        return self.metrics

    def get_metrics(self):
        """
        Retourne les métriques calculées (après appel à evaluate()).
        """
        if self.metrics is None:
            raise RuntimeError("Appelle d'abord evaluate() avant get_metrics().")
        return self.metrics

    def compare_models(self, models_dict, X_test, y_test):
        """
        Compare plusieurs modèles sur les mêmes données de test.
        
        :param models_dict: dictionnaire {nom: modèle}
        :param X_test: features de test
        :param y_test: vraies étiquettes
        :return: dictionnaire des résultats par modèle
        """
        print("\n" + "=" * 60)
        print("🏆 COMPARAISON DE MODÈLES")
        print("=" * 60)
        
        results = {}
        for name, model in models_dict.items():
            print(f"\n➡️  Modèle : {name}")
            evaluator = Evaluator(model, target_names=self.target_names)
            metrics = evaluator.evaluate(X_test, y_test, verbose=False)
            results[name] = metrics
            
            print(f"   Accuracy : {metrics['accuracy']:.4f} | "
                  f"F1-Score : {metrics['f1_score']:.4f}")
        
        # Trouver le meilleur modèle (selon F1-Score)
        best_model = max(results.items(), key=lambda x: x[1]["f1_score"])
        print("\n" + "-" * 60)
        print(f"🥇 Meilleur modèle (F1-Score) : {best_model[0]} "
              f"avec F1={best_model[1]['f1_score']:.4f}")
        print("=" * 60)
        
        return results