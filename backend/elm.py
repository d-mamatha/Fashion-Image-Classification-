import numpy as np

class ELM:
    def __init__(self, input_dim=None, hidden_dim=None, W=None, b=None, beta=None):
        # For training
        if input_dim is not None and hidden_dim is not None:
            self.W = np.random.randn(input_dim, hidden_dim)
            self.b = np.random.randn(hidden_dim)
            self.beta = None

        # For inference (loading model)
        else:
            self.W = W
            self.b = b
            self.beta = beta

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        H = self._sigmoid(X @ self.W + self.b)
        self.beta = np.linalg.pinv(H) @ y

    def predict(self, X):
        H = self._sigmoid(X @ self.W + self.b)
        return H @ self.beta
