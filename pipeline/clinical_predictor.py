"""
pipeline/clinical_predictor.py
Responsabilité : effectuer un diagnostic clinique à partir d'un modèle IA entraîné.
"""

import numpy as np
import pandas as pd


class ClinicalPredictor:
    """
    Classe qui utilise un modèle IA déjà entraîné pour diagnostiquer un patient.
    Retourne "Infecté" si la probabilité prédite >= 0.5, "Sain" sinon.
    """

    def __init__(self, trained_model, preprocessor=None):
        """
        Reçoit un modèle déjà entraîné et son preprocessor.
        """
        if trained_model is None:
            raise ValueError("Le modèle ne peut pas être None.")
        
        self.model = trained_model
        self.preprocessor = preprocessor

    def _prepare_data(self, patient_data):
        """
        Prépare les données pour la prédiction.
        Convertit les listes en DataFrame avec les bons noms de colonnes.
        """
        # Si c'est déjà un array numpy prétraité, le retourner tel quel
        if isinstance(patient_data, np.ndarray) and patient_data.shape[1] == 6:
            # Vérifier si c'est déjà standardisé (moyenne ≈ 0)
            if abs(patient_data.mean()) < 0.1:
                return patient_data
        
        # Si on a un preprocessor et des données brutes
        if self.preprocessor is not None:
            # Convertir en DataFrame si nécessaire
            if isinstance(patient_data, (list, np.ndarray)):
                feature_names = self.preprocessor.get_feature_names()
                patient_data = pd.DataFrame(patient_data, columns=feature_names)
            
            # Transformer avec le preprocessor
            return self.preprocessor.transform(patient_data)
        
        return patient_data

    def diagnose(self, patient_data):
        """
        Effectue le diagnostic d'un patient.
        
        :param patient_data: données du patient (déjà prétraitées ou brutes)
        :return: "Infecté" si probabilité >= 0.5, "Sain" sinon
        """
        patient_data = self._prepare_data(patient_data)
        
        # Récupérer les probabilités
        probabilities = self.model.predict_proba(patient_data)
        
        # Extraire la probabilité de la classe "Infecté" (index 1)
        prob_infection = probabilities[0][1]
        
        # Appliquer le seuil de décision
        if prob_infection >= 0.5:
            return "Infecté"
        else:
            return "Sain"

    def diagnose_with_details(self, patient_data):
        """
        Version enrichie qui retourne aussi la probabilité.
        """
        patient_data = self._prepare_data(patient_data)
        
        probabilities = self.model.predict_proba(patient_data)
        prob_infection = probabilities[0][1]
        
        diagnosis = "Infecté" if prob_infection >= 0.5 else "Sain"
        
        return {
            "diagnostic": diagnosis,
            "probabilite_infection": round(prob_infection, 4),
            "probabilite_sain": round(1 - prob_infection, 4)
        }