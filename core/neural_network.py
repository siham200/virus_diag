from sklearn.neural_network import MLPClassifier
from core.model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]