from sklearn.metrics import accuracy_score, classification_report
from core.dataset import VirusDataset

class Evaluator:
    def __init__(self, model):
        self.model = model
        self.dataset = VirusDataset()

    def evaluate(self):
        X_train, X_test, y_train, y_test = self.dataset.split_data()
        predictions = self.model.predict(X_test)
        # Convertir les probabilités en classes binaires
        binary_preds = (predictions >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, binary_preds)
        print(f" Précision : {accuracy:.2f}")
        print(classification_report(y_test, binary_preds))
        return accuracy