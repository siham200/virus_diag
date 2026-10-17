from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Classe abstraite de base pour tous les modèles d'IA du projet.
    Elle impose un contrat : tout modèle doit implémenter train() et predict().
    """

    @abstractmethod
    def train(self, X, y):
        """
        Entraîne le modèle sur les données X (features) et y (cible).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Fait des prédictions sur les données X.
        Retourne généralement des probabilités ou des classes.
        """
        pass