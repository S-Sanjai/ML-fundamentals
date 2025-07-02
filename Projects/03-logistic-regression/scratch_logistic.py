import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, l2_lambda=0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.loss_history = []
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        base_loss = - (1/m) * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
        l2_term = (self.l2_lambda / (2 * m)) * np.sum(np.square(self.w))
        return base_loss + l2_term

    def compute_gradients(self, X, y, y_hat):
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (y_hat - y)) + (self.l2_lambda / m) * self.w
        db = (1/m) * np.sum(y_hat - y)
        return dw, db

    def train(self, X, y):
        n_features = X.shape[1]
        self.w = np.zeros((n_features, 1))
        self.b = 0
        self.loss_history = []

        for i in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            y_hat = self.sigmoid(z)
            loss = self.compute_loss(y, y_hat)
            dw, db = self.compute_gradients(X, y, y_hat)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 100 == 0 or i == self.epochs - 1:
                self.loss_history.append(loss)

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call `.train()` first.")
        y_hat = self.sigmoid(np.dot(X, self.w) + self.b)
        return (y_hat > 0.5).astype(int)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return f"Accuracy: {accuracy * 100:.2f}%"

    def plot_loss(self):
        plt.figure(figsize=(8,5))
        plt.plot(range(0, len(self.loss_history)*100, 100), self.loss_history, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def stratified_split(X, y, test_size = 0.2, random_state = 42):
    np.random.seed(random_state)

    # Get unique classes and their indices
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []

    for class_label in unique_classes:
        # Get indices for this class
        class_indices = np.where(y == class_label)[0]

        # Shuffle indices for this class
        np.random.shuffle(class_indices)
                
        # Split this class
        n_test_class = int(len(class_indices) * test_size)
        test_indices.extend(class_indices[:n_test_class])
        train_indices.extend(class_indices[n_test_class:])

    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    

