from core.dataset import VirusDataset

class Trainer:
    def __init__(self, model):
        self.model = model
        self.dataset = VirusDataset()

    def train(self):
        X_train, X_test, y_train, y_test = self.dataset.split_data()
        self.model.train(X_train, y_train)
        print("✅ Modèle entraîné avec succès !")
        return self.model