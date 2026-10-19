"""
main.py
Point d'entrée principal du projet virus_diag.
"""

import os
import pickle
import numpy as np

from core.logistic_regression import LogisticRegressionModel
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.clinical_predictor import ClinicalPredictor


def main():
    print("=" * 70)
    print("🏥 SYSTÈME DE DIAGNOSTIC CLINIQUE - VIRUS_DIAG")
    print("=" * 70)
    
    data_path = "data/patient_data.csv"
    if not os.path.exists(data_path):
        print(f"\n❌ Erreur : Le fichier {data_path} n'existe pas.")
        return
    
    # ------------------------------------------------------------------
    # ÉTAPE 1 : Entraîner le modèle
    # ------------------------------------------------------------------
    print("\n📌 ÉTAPE 1 : Entraînement du modèle")
    print("-" * 70)
    
    model = LogisticRegressionModel(C=1.0, max_iter=1000, random_state=42)
    
    trainer = Trainer(
        model=model,
        data_path=data_path,
        test_size=0.2,
        random_state=42
    )
    
    trained_model = trainer.run(
        target_column="infected",
        save_path="models/trained_model.pkl"
    )
    
    X_test = trainer.X_test
    y_test = trainer.y_test
    preprocessor = trainer.preprocessor  # ⭐ Récupérer le preprocessor
    
    # ------------------------------------------------------------------
    # ÉTAPE 2 : Évaluer le modèle
    # ------------------------------------------------------------------
    print("\n📌 ÉTAPE 2 : Évaluation du modèle")
    print("-" * 70)
    
    evaluator = Evaluator(
        model=trained_model,
        target_names=["Sain", "Infecté"]
    )
    
    metrics = evaluator.evaluate(X_test, y_test, verbose=True)
    
    # ------------------------------------------------------------------
    # ÉTAPE 3 : Utiliser ClinicalPredictor
    # ------------------------------------------------------------------
    print("\n📌 ÉTAPE 3 : Diagnostic clinique avec ClinicalPredictor")
    print("-" * 70)
    
    # Créer le predictor avec le modèle ET le preprocessor
    predictor = ClinicalPredictor(
        trained_model=trained_model,
        preprocessor=preprocessor
    )
    
    # Test 1 : Patient depuis les données de test (déjà prétraitées)
    print("\n🔬 Test 1 : Patient depuis les données de test")
    patient_sample = X_test[0:1]
    diagnosis = predictor.diagnose(patient_sample)
    print(f"   → Diagnostic : {diagnosis}")
    
    details = predictor.diagnose_with_details(patient_sample)
    print(f"   → Probabilité d'infection : {details['probabilite_infection']:.2%}")
    
    # Test 2 : NOUVEAU patient (données brutes)
    print("\n🔬 Test 2 : Nouveau patient (données brutes)")
    # [age, temperature, toux, fatigue, mal_de_gorge, congestion]
    nouveau_patient = [[45, 38.5, 1, 8, 1, 1]]
    diagnosis_new = predictor.diagnose(nouveau_patient)
    print(f"   → Patient : {nouveau_patient[0]}")
    print(f"   → Diagnostic : {diagnosis_new}")
    
    details_new = predictor.diagnose_with_details(nouveau_patient)
    print(f"   → Probabilité d'infection : {details_new['probabilite_infection']:.2%}")
    
    # Test 3 : Patient sain
    print("\n🔬 Test 3 : Patient sain (données brutes)")
    patient_sain = [[25, 37.0, 0, 2, 0, 0]]
    diagnosis_sain = predictor.diagnose(patient_sain)
    print(f"   → Patient : {patient_sain[0]}")
    print(f"   → Diagnostic : {diagnosis_sain}")
    
    # ------------------------------------------------------------------
    # ÉTAPE 4 : Charger un modèle sauvegardé
    # ------------------------------------------------------------------
    print("\n📌 ÉTAPE 4 : Chargement d'un modèle sauvegardé")
    print("-" * 70)
    
    model_path = "models/trained_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        
        loaded_model = bundle["model"]
        loaded_preprocessor = bundle["preprocessor"]
        
        print(f"✅ Bundle chargé depuis : {model_path}")
        print(f"   - Modèle : {loaded_model.__class__.__name__}")
        print(f"   - Preprocessor : {loaded_preprocessor.__class__.__name__}")
        
        # Réutiliser avec ClinicalPredictor
        predictor_loaded = ClinicalPredictor(
            trained_model=loaded_model,
            preprocessor=loaded_preprocessor
        )
        
        # Tester avec un nouveau patient
        test_patient = [[50, 39.0, 1, 9, 1, 1]]
        diagnosis_loaded = predictor_loaded.diagnose(test_patient)
        print(f"   → Diagnostic patient [50, 39.0, 1, 9, 1, 1] : {diagnosis_loaded}")
    
    # ------------------------------------------------------------------
    # FIN
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ Pipeline terminé avec succès !")
    print("=" * 70)
    
    print("\n📊 RÉSUMÉ DES PERFORMANCES :")
    print(f"   Accuracy  : {metrics['accuracy']:.2%}")
    print(f"   Precision : {metrics['precision']:.2%}")
    print(f"   Recall    : {metrics['recall']:.2%}")
    print(f"   F1-Score  : {metrics['f1_score']:.2%}")


if __name__ == "__main__":
    main()