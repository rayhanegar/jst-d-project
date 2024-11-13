import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from scipy.special import softmax

class BackpropagationModel(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_conf, max_epoch=1000, max_error=0.1, learn_rate=0.5, print_per_epoch=100):
        self.layer_conf = layer_conf
        self.max_epoch = max_epoch
        self.max_error = max_error
        self.learn_rate = learn_rate
        self.print_per_epoch = print_per_epoch
        self.w = None
        self.epoch = 0
        self.mse = 1
        self.classes_ = None

    def _initialize_weights(self):
        np.random.seed(1)
        self.w = [np.random.rand(self.layer_conf[i] + 1, self.layer_conf[i+1]) * 0.1 
                  for i in range(len(self.layer_conf) - 1)]

    @staticmethod
    def sig(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def sigd(X):
        s = BackpropagationModel.sig(X)
        return s * (1 - s)

    def _softmax(self, x):
        return softmax(x, axis=0)

    def bp_fit(self, X, target):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term to input
        self.classes_ = np.unique(target)  # Unique classes in the target
        is_binary = len(self.classes_) == 2  # Determine binary vs. multi-class

        # One-hot encode target for multi-class; for binary, retain original 0/1 labels
        y = np.eye(len(self.classes_))[target] if not is_binary else target.reshape(-1, 1)

        self._initialize_weights()
        epoch, mse = 0, 1

        while (self.max_epoch == -1 or epoch < self.max_epoch) and mse > self.max_error:
            epoch += 1
            mse = 0
            
            for r in range(len(X)):
                # Forward pass
                n = [X[r]]
                for L in range(len(self.w)):
                    activation = np.dot(n[L], self.w[L])
                    # Apply sigmoid for hidden layers, and use appropriate activation for output
                    layer_output = (self.sig(activation) if L < len(self.w) - 1 
                                    else (self.sig(activation) if is_binary else self._softmax(activation)))
                    n.append(np.append(layer_output, 1) if L < len(self.w) - 1 else layer_output)
                
                # Calculate error and MSE
                e = y[r] - n[-1]
                mse += np.sum(e ** 2)

                # Determine delta based on classification type
                if is_binary:
                    d = e * self.sigd(np.dot(n[-2], self.w[-1]))  # Binary cross-entropy derivative
                else:
                    d = e  # Cross-entropy derivative for multi-class softmax

                # Backward pass with weight updates
                for L in range(len(self.w) - 1, -1, -1):
                    dw = self.learn_rate * np.outer(n[L], d)
                    self.w[L] += dw
                    if L > 0:
                        d = np.dot(d, self.w[L][:-1].T) * self.sigd(np.dot(n[L-1], self.w[L-1]))

            mse /= len(X)
            if self.print_per_epoch > -1 and epoch % self.print_per_epoch == 0:
                print(f'Epoch {epoch}, MSE: {mse:.6f}')

        self.epoch = epoch
        self.mse = mse

    def bp_predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term to input
        predictions = []

        for x in X:
            n = x
            for layer_weights in self.w:
                activation = np.dot(n, layer_weights)
                # Apply appropriate activation in the output layer
                n = (self.sig(activation) if layer_weights is not self.w[-1] 
                     else (self.sig(activation) if len(self.classes_) == 2 else self._softmax(activation)))
                n = np.append(n, 1) if layer_weights is not self.w[-1] else n
            predictions.append(n)

        return np.array(predictions)

    def fit(self, X, y):
        self.bp_fit(X, y)
        return self

    def predict(self, X):
        predictions = self.bp_predict(X)
        # For binary, threshold at 0.5; for multi-class, take argmax
        return (predictions > 0.5).astype(int) if len(self.classes_) == 2 else np.argmax(predictions, axis=1)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_params(self, deep=True):
        return {
            'layer_conf': self.layer_conf,
            'max_epoch': self.max_epoch,
            'max_error': self.max_error,
            'learn_rate': self.learn_rate,
            'print_per_epoch': self.print_per_epoch
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self