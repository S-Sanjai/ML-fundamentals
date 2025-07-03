"""
Logistic Regression Implementation from Scratch

This module provides a complete implementation of logistic regression with L2 regularization,
including training, prediction, evaluation, and visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegression:
    """
    Logistic Regression classifier with L2 regularization.
    
    This implementation uses gradient descent optimization to learn the optimal
    weights and bias for binary classification tasks.
    
    Parameters:
    -----------
    lr : float, default=0.01
        Learning rate for gradient descent optimization.
    epochs : int, default=1000
        Number of training iterations.
    l2_lambda : float, default=0.0
        L2 regularization strength. Higher values prevent overfitting.
        
    Attributes:
    -----------
    w : ndarray of shape (n_features, 1)
        Learned feature weights.
    b : float
        Learned bias term.
    loss_history : list
        Training loss values recorded every 100 epochs.
    """
    
    def __init__(self, lr=0.01, epochs=1000, l2_lambda=0.0):
        """Initialize the logistic regression model with hyperparameters."""
        self.lr = lr
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.loss_history = []
        self.w = None
        self.b = None

    def sigmoid(self, z):
        """
        Apply sigmoid activation function.
        
        Maps any real number to a value between 0 and 1 for probability estimation.
        
        Parameters:
        -----------
        z : ndarray
            Linear combination of features and weights (X·w + b).
            
        Returns:
        --------
        ndarray
            Sigmoid activation values between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_hat):
        """
        Compute the logistic regression loss with L2 regularization.
        
        Combines cross-entropy loss for classification with L2 regularization
        to prevent overfitting.
        
        Parameters:
        -----------
        y : ndarray
            True binary labels (0 or 1).
        y_hat : ndarray
            Predicted probabilities from sigmoid function.
            
        Returns:
        --------
        float
            Total loss value (cross-entropy + L2 regularization).
        """
        m = y.shape[0]  # Number of training examples
        
        # Cross-entropy loss with small epsilon to prevent log(0)
        base_loss = - (1/m) * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
        
        # L2 regularization term (penalty on large weights)
        l2_term = (self.l2_lambda / (2 * m)) * np.sum(np.square(self.w))
        
        return base_loss + l2_term

    def compute_gradients(self, X, y, y_hat):
        """
        Compute gradients of the loss function with respect to weights and bias.
        
        Uses the chain rule to compute partial derivatives for gradient descent.
        
        Parameters:
        -----------
        X : ndarray of shape (m, n_features)
            Feature matrix.
        y : ndarray
            True binary labels.
        y_hat : ndarray
            Predicted probabilities.
            
        Returns:
        --------
        tuple
            (dw, db) - gradients for weights and bias respectively.
        """
        m = X.shape[0]  # Number of training examples
        
        # Gradient with respect to weights (including L2 regularization)
        dw = (1/m) * np.dot(X.T, (y_hat - y)) + (self.l2_lambda / m) * self.w
        
        # Gradient with respect to bias (no regularization applied to bias)
        db = (1/m) * np.sum(y_hat - y)
        
        return dw, db

    def train(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Implements the complete training loop with parameter initialization,
        forward pass, loss computation, and parameter updates.
        
        Parameters:
        -----------
        X : ndarray of shape (m, n_features)
            Training feature matrix.
        y : ndarray of shape (m,)
            Training labels (binary: 0 or 1).
        """
        # Initialize parameters
        n_features = X.shape[1]
        self.w = np.zeros((n_features, 1))  # Initialize weights to zero
        self.b = 0  # Initialize bias to zero
        self.loss_history = []  # Reset loss history

        # Training loop
        for i in range(self.epochs):
            # Forward pass: compute predictions
            z = np.dot(X, self.w) + self.b  # Linear combination
            y_hat = self.sigmoid(z)  # Apply sigmoid activation
            
            # Compute loss
            loss = self.compute_loss(y, y_hat)
            
            # Backward pass: compute gradients
            dw, db = self.compute_gradients(X, y, y_hat)

            # Update parameters using gradient descent
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Record loss every 100 epochs for monitoring convergence
            if i % 100 == 0 or i == self.epochs - 1:
                self.loss_history.append(loss)

    def predict(self, X):
        """
        Make binary predictions on new data.
        
        Uses the trained model to predict class labels by applying
        a threshold of 0.5 to predicted probabilities.
        
        Parameters:
        -----------
        X : ndarray of shape (m, n_features)
            Feature matrix for prediction.
            
        Returns:
        --------
        ndarray
            Predicted binary labels (0 or 1).
            
        Raises:
        -------
        ValueError
            If the model hasn't been trained yet.
        """
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call `.train()` first.")
        
        # Compute probabilities and apply threshold
        y_hat = self.sigmoid(np.dot(X, self.w) + self.b)
        return (y_hat > 0.5).astype(int)

    def evaluate(self, X, y):
        """
        Evaluate model performance using accuracy metric.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix for evaluation.
        y : ndarray
            True binary labels.
            
        Returns:
        --------
        str
            Formatted accuracy percentage string.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return f"Accuracy: {accuracy * 100:.2f}%"

    def plot_loss(self):
        """
        Plot the training loss curve over epochs.
        
        Visualizes how the loss decreases during training to assess
        convergence and detect potential overfitting issues.
        """
        plt.figure(figsize=(8,5))
        plt.plot(range(0, len(self.loss_history)*100, 100), self.loss_history, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, X, y):
        """
        Computes and plots the confusion matrix.

        A confusion matrix is a table used to evaluate the performance of a
        classification model. It shows the number of true positives, true negatives,
        false positives, and false negatives.

        Parameters:
        -----------
        X : ndarray
            Feature matrix for evaluation.
        y : ndarray
            True binary labels.
        """
        y_pred = self.predict(X)
        
        # Ensure y is a 1D array
        if y.ndim > 1:
            y = y.ravel()
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        # Compute confusion matrix components
        tp = np.sum((y == 1) & (y_pred == 1))
        tn = np.sum((y == 0) & (y_pred == 0))
        fp = np.sum((y == 0) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plotting the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malignant'], 
                    yticklabels=['Benign', 'Malignant'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()


def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split to maintain class distribution.
    
    Ensures that both training and test sets have approximately the same
    proportion of samples from each class as the original dataset.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix to split.
    y : ndarray of shape (n_samples,)
        Target labels to split.
    test_size : float, default=0.2
        Proportion of dataset to include in test split.
    random_state : int, default=42
        Random seed for reproducible results.
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) - stratified splits of the data.
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Get unique classes and their indices
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []

    # Process each class separately to maintain stratification
    for class_label in unique_classes:
        # Get indices for this class
        class_indices = np.where(y == class_label)[0]

        # Shuffle indices for this class
        np.random.shuffle(class_indices)
                
        # Split this class according to test_size ratio
        n_test_class = int(len(class_indices) * test_size)
        test_indices.extend(class_indices[:n_test_class])
        train_indices.extend(class_indices[n_test_class:])

    # Convert to arrays and shuffle to avoid any ordering bias
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Return the stratified splits
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]