from sklearn.linear_model import LogisticRegression
from core.model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probabilité d'infection