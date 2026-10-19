"""
core/logistic_regression.py
Implémentation d'un modèle de Régression Logistique compatible avec la structure du projet.
"""

from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticRegressionModel:
    """
    Wrapper autour de la régression logistique de scikit-learn.
    Cette classe expose les méthodes fit(), predict() et predict_proba()
    pour être utilisée par Trainer et ClinicalPredictor.
    """

    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        """
        Initialise le modèle.
        
        :param C: Inverse de la force de régularisation (plus petit = plus fort).
        :param max_iter: Nombre maximum d'itérations pour la convergence.
        :param random_state: Graine aléatoire pour la reproductibilité.
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'  # Solveur robuste pour les petits/moyens jeux de données
        )
        self.is_trained = False

    def fit(self, X, y):
        """
        Entraîne le modèle sur les données fournies.
        
        :param X: Matrice des features (numpy array ou DataFrame)
        :param y: Vecteur des cibles (labels)
        """
        print("🔄 Entraînement de la Régression Logistique...")
        self.model.fit(X, y)
        self.is_trained = True
        print("✅ Modèle entraîné.")

    def predict(self, X):
        """
        Prédit la classe directe (0 ou 1).
        
        :param X: Données d'entrée
        :return: Tableau des classes prédites
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné. Appelez fit() d'abord.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Prédit les probabilités pour chaque classe.
        
        :param X: Données d'entrée
        :return: Tableau de shape (n_samples, 2) avec [prob_sain, prob_infecte]
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné. Appelez fit() d'abord.")
        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        Retourne la précision moyenne sur les données de test.
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        return self.model.score(X, y)