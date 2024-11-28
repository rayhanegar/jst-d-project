import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from scipy.special import softmax

class BackpropagationModelv2(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_conf, max_epoch=1000, max_error=0.1, learn_rate=0.5, print_per_epoch=100, batch_size=1):
        if not isinstance(layer_conf, list) or not all(isinstance(i, int) for i in layer_conf):
            raise ValueError("layer_conf must be a list of integers")
        if max_epoch <= 0 and max_epoch != -1:
            raise ValueError("max_epoch must be a positive integer or -1 for unlimited epochs")
        if max_error <= 0:
            raise ValueError("max_error must be a positive float")
        if learn_rate <= 0:
            raise ValueError("learn_rate must be a positive float")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.layer_conf = layer_conf
        self.max_epoch = max_epoch
        self.max_error = max_error
        self.learn_rate = learn_rate
        self.print_per_epoch = print_per_epoch
        self.batch_size = batch_size
        self.w = None
        self.epoch = 0
        self.mse = 1
        self.classes_ = None

    def _initialize_weights(self):
        np.random.seed(1)
        self.w = [np.random.randn(self.layer_conf[i] + 1, self.layer_conf[i+1]) * np.sqrt(2 / (self.layer_conf[i] + 1))
                  for i in range(len(self.layer_conf) - 1)]

    @staticmethod
    def sig(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def sigd(X):
        s = BackpropagationModelv2.sig(X)
        return s * (1 - s)

    def _softmax(self, x):
        return softmax(x, axis=0)

    def bp_fit(self, X, target):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  
        self.classes_ = np.unique(target)  
        is_binary = len(self.classes_) == 2  

        y = np.eye(len(self.classes_))[target] if not is_binary else target.reshape(-1, 1)

        self._initialize_weights()
        epoch, mse = 0, 1

        while (self.max_epoch == -1 or epoch < self.max_epoch) and mse > self.max_error:
            epoch += 1
            mse = 0
            for start in range(0, len(X), self.batch_size):
                end = start + self.batch_size
                batch_X, batch_y = X[start:end], y[start:end]
                batch_mse = 0
                batch_dw = [np.zeros_like(w) for w in self.w]
                
                for r in range(len(batch_X)):
                    # Forward pass
                    n = [batch_X[r]]
                    for L in range(len(self.w)):
                        activation = np.dot(n[L], self.w[L])
                        layer_output = (self.sig(activation) if L < len(self.w) - 1 
                                        else (self.sig(activation) if is_binary else self._softmax(activation)))
                        n.append(np.append(layer_output, 1) if L < len(self.w) - 1 else layer_output)
                    
                    # Calculate error and MSE
                    e = batch_y[r] - n[-1]
                    batch_mse += np.sum(e ** 2)

                    # Determine delta based on classification type
                    if is_binary:
                        d = e * self.sigd(np.dot(n[-2], self.w[-1]))
                    else:
                        d = e

                    # Backward pass with weight updates
                    for L in range(len(self.w) - 1, -1, -1):
                        dw = self.learn_rate * np.outer(n[L], d)
                        batch_dw[L] += dw
                        if L > 0:
                            d = np.dot(d, self.w[L][:-1].T) * self.sigd(np.dot(n[L-1], self.w[L-1]))

                # Update weights after processing the batch
                for L in range(len(self.w)):
                    self.w[L] += batch_dw[L] / self.batch_size

                mse += batch_mse / self.batch_size

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
            'print_per_epoch': self.print_per_epoch,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self