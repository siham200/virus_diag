
"""
core/random_forest.py

Implémentation d'un modèle Random Forest
compatible avec le pipeline virus_diag.
"""

from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        random_state=42
    ):
        """
        Initialise le modèle Random Forest.

        Parameters:
            n_estimators : nombre d'arbres
            max_depth : profondeur maximale des arbres
            random_state : graine aléatoire
        """

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

        self.is_trained = False

    def fit(self, X, y):
        """Entraîne le modèle."""

        print("Entrainement du modele Random Forest...")

        self.model.fit(X, y)
        self.is_trained = True

        print("Modele Random Forest entraine avec succes.")

    def predict(self, X):
        """Prédit les classes."""

        if not self.is_trained:
            raise RuntimeError(
                "Le modele doit etre entraine avant prediction."
            )

        return self.model.predict(X)

    def predict_proba(self, X):
        """Retourne les probabilités de classification."""

        if not self.is_trained:
            raise RuntimeError(
                "Le modele doit etre entraine avant prediction."
            )

        return self.model.predict_proba(X)

    def score(self, X, y):
        """Calcule la précision du modèle."""

        if not self.is_trained:
            raise RuntimeError(
                "Le modele doit etre entraine avant evaluation."
            )

        return self.model.score(X, y)
