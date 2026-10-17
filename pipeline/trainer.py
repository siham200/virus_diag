import joblib
import os
from core.dataset import VirusDataset
from utils.preprocessing import DataPreprocessor

class Trainer:
    """
    Classe responsable de l'orchestration de l'entraînement d'un modèle.
    """

    def __init__(self, model, preprocessor=None):
        """
        :param model: Une instance d'un modèle respectant BaseModel
        :param preprocessor: Une instance de DataPreprocessor (optionnel, un par défaut sera créé)
        """
        self.model = model
        self.preprocessor = preprocessor if preprocessor else DataPreprocessor()
        self.dataset = VirusDataset(filepath="data/patient_data.csv")

    def train(self, save_model=True, model_path="models/model.pkl", preprocessor_path="models/preprocessor.pkl"):
        """
        Méthode principale qui exécute tout le pipeline d'entraînement.
        """
        print("🚀 Début de l'entraînement...")

        # 1. Chargement et division des données
        print("📥 Chargement des données...")
        X_train, X_test, y_train, y_test = self.dataset.split_data(test_size=0.2)

        # 2. Prétraitement des données (FIT sur le train uniquement !)
        print("⚙️ Prétraitement des données...")
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test) # Juste transform pour le test

        # 3. Entraînement du modèle
        print("🧠 Entraînement du modèle...")
        self.model.train(X_train_scaled, y_train)

        # 4. Sauvegarde (Optionnel mais recommandé pour la production)
        if save_model:
            # Créer le dossier models s'il n'existe pas
            os.makedirs("models", exist_ok=True)
            
            # Sauvegarder le modèle et le préprocesseur
            joblib.dump(self.model, model_path)
            joblib.dump(self.preprocessor, preprocessor_path)
            print(f"💾 Modèle sauvegardé dans : {model_path}")
            print(f"💾 Préprocesseur sauvegardé dans : {preprocessor_path}")

        print("✅ Entraînement terminé avec succès !\n")
        
        # Retourner les données de test pour permettre l'évaluation immédiate
        return X_test_scaled, y_test