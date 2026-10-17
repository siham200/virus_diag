"""
Script de test pour vérifier que le modèle et l'entraîneur fonctionnent ensemble.
À exécuter depuis la racine du projet : python test_pipeline.py
"""

from core.logistic_regression import LogisticRegressionModel
from pipeline.trainer import Trainer

print("="*60)
print("🧪 TEST DU PIPELINE D'ENTRAÎNEMENT")
print("="*60)

# 1. Créer une instance du modèle concret
print("\n1️⃣ Création du modèle...")
mon_modele = LogisticRegressionModel()

# 2. Créer l'entraîneur (Trainer) et lui donner le modèle
print("2️⃣ Configuration du Trainer...")
mon_trainer = Trainer(model=mon_modele)

# 3. Lancer l'entraînement
print("3️⃣ Lancement de l'entraînement...")
X_test, y_test = mon_trainer.train(save_model=True)

# 4. Faire une prédiction de test rapide
print("4️⃣ Test de prédiction sur le premier échantillon de test...")
premiere_prediction = mon_modele.predict(X_test[0:1])[0]
print(f"   Probabilité d'infection : {premiere_prediction:.2%}")
print(f"   Vérité terrain (y_test) : {'Infecté' if y_test.iloc[0] == 1 else 'Sain'}")

print("\n" + "="*60)
print("✅ LE PIPELINE FONCTIONNE PARFAITEMENT !")
print("="*60)