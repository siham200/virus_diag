import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Classe de prétraitement des données pour le projet virus_diag.
    
    Elle gère :
    - La standardisation des features numériques
    - L'imputation des valeurs manquantes
    - La séparation fit/transform pour éviter la fuite de données
    """

    def __init__(self):
        """
        Initialise le préprocesseur avec un scaler et un imputer.
        """
        # StandardScaler : centre (moyenne=0) et réduit (écart-type=1) les données
        self.scaler = StandardScaler()
        
        # SimpleImputer : remplace les valeurs manquantes par la moyenne
        self.imputer = SimpleImputer(strategy='mean')
        
        # Pour savoir si le preprocessor a été "ajusté" (fitted)
        self.is_fitted = False

    def fit(self, X):
        """
        Apprend les paramètres de transformation sur les données d'entraînement.
        À n'appeler qu'UNE SEULE FOIS sur les données de train.
        
        :param X: DataFrame ou array des features d'entraînement
        """
        # 1. Ajuster l'imputer (calcule la moyenne de chaque colonne)
        self.imputer.fit(X)
        
        # 2. Imputer les valeurs manquantes
        X_imputed = self.imputer.transform(X)
        
        # 3. Ajuster le scaler (calcule moyenne et écart-type)
        self.scaler.fit(X_imputed)
        
        self.is_fitted = True
        print("✅ Preprocessor ajusté (fit) sur les données d'entraînement")
        return self

    def transform(self, X):
        """
        Applique la transformation aux données (train OU test OU nouveau patient).
        
        :param X: DataFrame ou array à transformer
        :return: Données transformées (numpy array)
        """
        if not self.is_fitted:
            raise RuntimeError("❌ Le preprocessor n'a pas été ajusté. Appelez fit() d'abord.")
        
        # 1. Imputer avec les moyennes apprises sur le train
        X_imputed = self.imputer.transform(X)
        
        # 2. Standardiser avec les paramètres appris sur le train
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled

    def fit_transform(self, X):
        """
        Combine fit() et transform() en une seule étape (pour les données de train).
        
        :param X: DataFrame ou array des features d'entraînement
        :return: Données transformées
        """
        self.fit(X)
        return self.transform(X)

    def transform_patient(self, patient_data):
        
        if not self.is_fitted:
            raise RuntimeError("❌ Le preprocessor n'a pas été ajusté.")
            
        # Convertir en DataFrame avec les bons noms de colonnes
        feature_names = self.get_feature_names()
        patient_array = np.array(patient_data).reshape(1, -1)
        patient_df = pd.DataFrame(patient_array, columns=feature_names)
            
            # Appliquer les mêmes transformations
        return self.transform(patient_df)
    def get_feature_names(self):
        """
        Retourne les noms des features.
        """
        return ["age", "temperature", "toux", "fatigue", "mal_de_gorge", "congestion"]


# ============================================
# 🔬 TEST DU FICHIER PREPROCESSING.PY
# ============================================
if __name__ == "__main__":
    from core.dataset import VirusDataset

    print("="*60)
    print("🧪 TEST DU PREPROCESSING")
    print("="*60)

    # 1. Charger les données
    print("\n📥 Étape 1 : Chargement du dataset")
    dataset = VirusDataset(filepath="data/patient_data.csv")
    X_train, X_test, y_train, y_test = dataset.split_data(test_size=0.2)

    # 2. Créer le preprocessor
    print("\n⚙️ Étape 2 : Création du DataPreprocessor")
    preprocessor = DataPreprocessor()

    # 3. Afficher les données AVANT preprocessing
    print("\n📊 Données AVANT preprocessing (5 premières lignes de X_train) :")
    print(X_train.head())
    print(f"\nMoyenne de chaque colonne :")
    print(X_train.mean())
    print(f"\nÉcart-type de chaque colonne :")
    print(X_train.std())

    # 4. Appliquer fit_transform sur les données de train
    print("\n Étape 3 : fit_transform sur X_train")
    X_train_scaled = preprocessor.fit_transform(X_train)
    print(f"Type de retour : {type(X_train_scaled)}")
    print(f"Shape : {X_train_scaled.shape}")

    # 5. Afficher les données APRÈS preprocessing
    print("\n📊 Données APRÈS preprocessing (5 premières lignes) :")
    print(X_train_scaled[:5])
    
    # Convertir en DataFrame pour afficher les stats
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=preprocessor.get_feature_names())
    print(f"\nMoyenne après standardisation (doit être ≈ 0) :")
    print(X_train_scaled_df.mean().round(6))
    print(f"\nÉcart-type après standardisation (doit être ≈ 1) :")
    print(X_train_scaled_df.std().round(6))

    # 6. Appliquer transform sur les données de test
    print("\n🔄 Étape 4 : transform sur X_test (sans refit)")
    X_test_scaled = preprocessor.transform(X_test)
    print(f"Shape de X_test_scaled : {X_test_scaled.shape}")

    # 7. Tester avec un nouveau patient
    print("\n🩺 Étape 5 : Prétraitement d'un nouveau patient")
    nouveau_patient = [45, 38.5, 1, 1, 1, 1]
    patient_scaled = preprocessor.transform_patient(nouveau_patient)
    print(f"Patient original  : {nouveau_patient}")
    print(f"Patient transformé: {patient_scaled[0].round(4)}")

    # 8. Tester avec un patient sain
    print("\n🩺 Étape 6 : Prétraitement d'un patient sain")
    patient_sain = [25, 37.2, 0, 1, 0, 0]
    patient_sain_scaled = preprocessor.transform_patient(patient_sain)
    print(f"Patient original  : {patient_sain}")
    print(f"Patient transformé: {patient_sain_scaled[0].round(4)}")

    # 9. Test d'erreur : transform sans fit
    print("\n⚠️ Étape 7 : Test de l'erreur si on appelle transform sans fit")
    try:
        nouveau_preprocessor = DataPreprocessor()
        nouveau_preprocessor.transform([[1, 2, 3, 4, 5, 6]])
    except RuntimeError as e:
        print(f"✅ Erreur correctement levée : {e}")

    print("\n" + "="*60)
    print("✅ TOUS LES TESTS DE PREPROCESSING SONT PASSÉS !")
    print("="*60)