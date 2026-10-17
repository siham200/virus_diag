import pandas as pd
from sklearn.model_selection import train_test_split


class VirusDataset:
    """
    Classe pour charger et préparer le dataset patient_data.csv
    """

    def __init__(self, filepath="data/patient_data.csv"):
        """
        Initialise le dataset en chargeant le fichier CSV
        :param filepath: Chemin vers le fichier CSV
        """
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        print(f"✅ Dataset chargé : {len(self.data)} lignes, {len(self.data.columns)} colonnes")

    def get_features_and_target(self):
        """
        Sépare les features (X) de la cible (y)
        :return: X (features), y (cible)
        """
        X = self.data.drop(columns=["infected"])
        y = self.data["infected"]
        return X, y

    def split_data(self, test_size=0.2, random_state=42):
        """
        Divise les données en train/test
        :param test_size: Proportion des données de test (défaut: 20%)
        :param random_state: Graine pour la reproductibilité
        :return: X_train, X_test, y_train, y_test
        """
        X, y = self.get_features_and_target()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"✅ Données divisées :")
        print(f"   - Train : {len(X_train)} échantillons")
        print(f"   - Test  : {len(X_test)} échantillons")
        return X_train, X_test, y_train, y_test

    def get_summary(self):
        """
        Affiche un résumé du dataset
        """
        print("\n" + "="*50)
        print("📊 RÉSUMÉ DU DATASET")
        print("="*50)
        print(f"Nombre total de patients : {len(self.data)}")
        print(f"Nombre de features       : {len(self.data.columns) - 1}")
        print(f"Colonnes                 : {list(self.data.columns)}")
        print(f"\nRépartition des classes :")
        print(self.data["infected"].value_counts())
        print(f"\nStatistiques descriptives :")
        print(self.data.describe())
        print("="*50 + "\n")


# ============================================
# 🔬 TEST DU FICHIER DATASET.PY
# ============================================
if __name__ == "__main__":

    print("🧪 TEST 1 : Chargement du dataset")
    print("-" * 40)
    dataset = VirusDataset(filepath="data/patient_data.csv")

    print("\n🧪 TEST 2 : Récupération des features et target")
    print("-" * 40)
    X, y = dataset.get_features_and_target()
    print(f"Shape de X (features) : {X.shape}")
    print(f"Shape de y (target)   : {y.shape}")
    print(f"\nPremières lignes de X :\n{X.head()}")
    print(f"\nPremières valeurs de y :\n{y.head()}")

    print("\n TEST 3 : Split train/test")
    print("-" * 40)
    X_train, X_test, y_train, y_test = dataset.split_data(test_size=0.2)
    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")

    print("\n🧪 TEST 4 : Résumé complet")
    print("-" * 40)
    dataset.get_summary()

    print("\n✅ TOUS LES TESTS SONT PASSÉS !")