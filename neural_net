import numpy as np


class NeuralNetwork:
    """Простая полносвязная нейросеть (MLP) на чистом numpy."""

    def __init__(self, layers, lr=0.5, seed=1):
        # layers, например [2, 4, 1]: 2 входа, 4 нейрона скрытого слоя, 1 выход
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W = [rng.standard_normal((layers[i], layers[i + 1])) * np.sqrt(1 / layers[i])
                  for i in range(len(layers) - 1)]
        self.b = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(a):
        # a — уже результат sigmoid
        return a * (1 - a)

    def forward(self, X):
        self.A = [X]
        for W, b in zip(self.W, self.b):
            self.A.append(self.sigmoid(self.A[-1] @ W + b))
        return self.A[-1]

    def backward(self, y):
        m = y.shape[0]
        delta = (self.A[-1] - y) * self.sigmoid_deriv(self.A[-1])
        for i in reversed(range(len(self.W))):
            dW = self.A[i].T @ delta / m
            db = delta.sum(axis=0, keepdims=True) / m
            if i > 0:
                delta = (delta @ self.W[i].T) * self.sigmoid_deriv(self.A[i])
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def train(self, X, y, epochs=10000, log_every=1000):
        for epoch in range(1, epochs + 1):
            out = self.forward(X)
            self.backward(y)
            if epoch % log_every == 0:
                loss = np.mean((y - out) ** 2)
                print(f"Эпоха {epoch}, ошибка: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


if __name__ == "__main__":
    # Обучаем сеть задаче XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    nn = NeuralNetwork([2, 4, 1], lr=0.5)
    nn.train(X, y, epochs=10000, log_every=1000)

    print("\nРезультат:")
    for x, p in zip(X, nn.predict(X)):
        print(f"{x} -> {p[0]:.3f} (округлено: {int(round(p[0]))})")
