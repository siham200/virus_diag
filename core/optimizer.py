from sklearn.model_selection import GridSearchCV

class ModelOptimizer:
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid

    def optimize(self, X, y):
        grid = GridSearchCV(self.model, self.param_grid, cv=3)
        grid.fit(X, y)
        return grid.best_estimator_