"""
pipeline/trainer.py
Responsabilité : entraîner un modèle d'IA à partir de données patients.
Utilise la classe DataPreprocessor de utils/preprocessing.py.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import de TA classe DataPreprocessor
from utils.preprocessing import DataPreprocessor


class Trainer:
    """
    Classe qui entraîne un modèle d'IA.
    Elle est agnostique du modèle utilisé (polymorphisme).
    """

    def __init__(self, model, data_path="data/patient_data.csv", 
                 test_size=0.2, random_state=42):
        """
        :param model: instance d'un modèle (LogisticRegression ou NeuralNetwork)
        :param data_path: chemin vers le fichier CSV
        :param test_size: proportion des données pour le test
        :param random_state: graine pour la reproductibilité
        """
        self.model = model
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        # Préprocesseur (TA classe)
        self.preprocessor = DataPreprocessor()

        # Résultats de l'entraînement
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # ------------------------------------------------------------------
    # 1. Charger les données
    # ------------------------------------------------------------------
    def load_data(self):
        """Charge le fichier CSV et retourne un DataFrame."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Fichier introuvable : {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        return df

    # ------------------------------------------------------------------
    # 2. Préparer les données (prétraitement + split)
    # ------------------------------------------------------------------
    def prepare_data(self, df, target_column="status"):
        """
        Sépare features / cible, fait le split train/test,
        puis ajuste le preprocessor sur le train uniquement.
        """
        if target_column not in df.columns:
            raise ValueError(f"La colonne cible '{target_column}' n'existe pas.")

        # Séparer X et y
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encoder la cible si c'est du texte
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoder = le
            print(f"   - Cible encodée : {le.classes_} → {list(range(len(le.classes_)))}")
        else:
            self.label_encoder = None

        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # ⚠️ IMPORTANT : fit le preprocessor sur le TRAIN uniquement
        print("🔧 Ajustement du preprocessor sur les données d'entraînement...")
        self.X_train = self.preprocessor.fit_transform(self.X_train)

        # Transform le test avec les paramètres du train (pas de fuite !)
        self.X_test = self.preprocessor.transform(self.X_test)

        print(f"✅ Données préparées : train={self.X_train.shape}, test={self.X_test.shape}")

    # ------------------------------------------------------------------
    # 3. Entraîner le modèle
    # ------------------------------------------------------------------
    def train(self):
        """
        Lance l'entraînement du modèle sur les données de train.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Appelle d'abord prepare_data() avant train().")

        print(f"🚀 Entraînement du modèle {self.model.__class__.__name__} en cours...")
        self.model.fit(self.X_train, self.y_train)
        print("✅ Modèle entraîné avec succès.")
        return self.model

    # ------------------------------------------------------------------
    # 4. Sauvegarder le modèle ET le preprocessor
    # ------------------------------------------------------------------
    def save_model(self, path="models/trained_model.pkl"):
        """
        Sauvegarde le modèle ET le preprocessor ensemble.
        Le preprocessor est nécessaire pour transformer de nouvelles données.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # On sauvegarde un dictionnaire contenant le modèle + le preprocessor
        bundle = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "label_encoder": self.label_encoder,
            "feature_names": self.preprocessor.get_feature_names()
        }
        
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"💾 Modèle + preprocessor sauvegardés dans : {path}")

    # ------------------------------------------------------------------
    # 5. Pipeline complet
    # ------------------------------------------------------------------
    def run(self, target_column="status", save_path="models/trained_model.pkl"):
        """
        Exécute tout le pipeline : chargement → préparation → entraînement → sauvegarde.
        """
        df = self.load_data()
        self.prepare_data(df, target_column=target_column)
        self.train()
        self.save_model(path=save_path)
        return self.model