import random

import numpy as np
from numpy.random import choice, seed
from numpy import ndarray
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class MyLogisticRegression(object):
    """Logistic regression class.
    Estimation function (Maximize the likelihood):
    z = WX + b
    y = 1 / (1 + e**(-z))
    Likelihood function:
    P(y | X, W, b) = y_hat^y * (1-y_hat)^(1-y)
    L = Product(P(y | X, W, b))
    Take the logarithm of both sides of this equation:
    log(L) = Sum(log(P(y | X, W, b)))
    log(L) = Sum(log(y_hat^y * (1-y_hat)^(1-y)))
    log(L) = Sum(y * log(y_hat) + (1-y) * log(1-y_hat)))
    Get partial derivative of W and b:
    1. dz/dW = X
    2. dy_hat/dz = y_hat * (1-y_hat)
    3. dlog(L)/dy_hat = y * 1/y_hat - (1-y) * 1/(1-y_hat)
    4. dz/db = 1
    According to 1,2,3:
    dlog(L)/dW = dlog(L)/dy_hat * dy_hat/dz * dz/dW
    dlog(L)/dW = (y - y_hat) * X
    According to 2,3,4:
    dlog(L)/db = dlog(L)/dy_hat * dy_hat/dz * dz/db
    dlog(L)/db = y - y_hat
    """

    def __init__(self):

        self.bias = None
        self.weights = None

        # define the method to calculate prediction of label in order to get gradient.
        if self.__class__.__name__ == "LogisticRegression":
            self._predict = self.predict_proba
        else:
            self._predict = self.predict

    def sigmoid(self, x):
        """Calculate the sigmoid value of x.
        Sigmoid(x) = 1 / (1 + e^(-x))
        It would cause math range error when x < -709
        """

        return 1 / (1 + np.exp(-x))

    def _get_gradient(self, data: ndarray, label: ndarray):
        """Calculate the gradient of the partial derivative.
        """

        y_hat = self._predict(data)

        # Calculate the gradient according to the dimention of data, label.
        grad_bias = label - y_hat
        if data.ndim == 1:
            grad_weights = grad_bias * data
        elif data.ndim == 2:
            grad_weights = grad_bias[:, None] * data
            grad_weights = grad_weights.mean(axis=0)
            grad_bias = grad_bias.mean()
        else:
            raise ValueError("Dimension of data has to be 1 or 2!")

        return grad_bias, grad_weights

    def _batch_gradient_descent(self, data: ndarray, label: ndarray, X_valid, y_valid, learning_rate: float, epochs: int, tol=1e-4):
        """Update the gradient by the whole dataset.
        b = b - learning_rate * 1/m * b_grad_i, b_grad_i <- grad
        W = W - learning_rate * 1/m * w_grad_i, w_grad_i <- grad
        """

        # Initialize the bias and weights.
        _, n_cols = data.shape
        self.bias = 0
        # self.weights = np.random.normal(size=n_cols)
        self.weights = np.zeros(n_cols)
        f1_list = []
        for i in range(epochs):
            # Calculate and sum the gradient delta of each sample.
            grad_bias, grad_weights = self._get_gradient(data, label)

            # Show the gradient of each epoch.
            grad = (grad_bias + grad_weights.mean()) / 2
            if np.abs(grad) <= tol:
                break

            lr_preds = self.predict(X_valid)
            f1 = f1_score(y_valid, lr_preds)

            f1_list.append(f1)
            if f1>0.45:
                break

            print("Epochs %d gradient %.3f" % (i + 1, grad), flush=True)
            # print("Epochs %d F1 Score %.3f" % (i + 1, f1), flush=True)

            # Update the bias and weight by gradient of current epoch
            self.bias += learning_rate * grad_bias
            self.weights += learning_rate * grad_weights


        plt.plot(range(len(f1_list)), f1_list)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('F1 Score', fontsize=16)
        plt.title('Training F1 score on Valid', fontsize=16)
        plt.legend(loc="lower right", fontsize=16)
        plt.show()



    def _stochastic_gradient_descent(self, data: ndarray, label: ndarray, X_valid, y_valid, learning_rate: float,
                                     epochs: int, sample_rate: float, random_state, tol=1e-4):
        """Update the gradient by the random sample of dataset.
        b = b - learning_rate * b_sample_grad_i, b_sample_grad_i <- sample_grad
        W = W - learning_rate * w_sample_grad_i, w_sample_grad_i <- sample_grad
        """

        # Set random state.
        if random_state is not None:
            seed(random_state)

        # Initialize the bias and weights.
        n_rows, n_cols = data.shape
        self.bias = 0
        self.weights = np.random.normal(size=n_cols)
        f1_list = []
        n_sample = int(n_rows * sample_rate)
        for i in range(epochs):
            for idx in choice(range(n_rows), n_sample, replace=False):
                # Calculate the gradient delta of each sample
                grad_bias, grad_weights = self._get_gradient(
                    data[idx], label[idx])

                # Update the bias and weight by gradient of current sample
                self.bias += learning_rate * grad_bias
                self.weights += learning_rate * grad_weights

            # Show the gradient of each epoch.
            grad_bias, grad_weights = self._get_gradient(data, label)
            grad = (grad_bias + grad_weights.mean()) / 2

            if np.abs(grad) <= tol:
                break

            lr_preds = self.predict(X_valid)
            f1 = f1_score(y_valid, lr_preds)
            f1_list.append(f1)
            if f1>0.45:
                break

            print("Epochs %d F1 Score %.3f" % (i + 1, f1), flush=True)
            # print("Epochs %d gradient %.3f" % (i + 1, grad), flush=True)

        plt.plot(range(len(f1_list)), f1_list)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('F1 Score', fontsize=16)
        plt.title('Training F1 score on Valid', fontsize=16)
        plt.legend(loc="lower right", fontsize=16)
        plt.show()

        # Cancel random state.
        if random_state is not None:
            seed(None)

    def fit(self, data: ndarray, label: ndarray, X_valid, y_valid, learning_rate=0.01, epochs=1000,
            method="batch", sample_rate=0.8, random_state=123, tol=1e-5):

        assert method in ("batch", "stochastic"), str(method)
        # Batch gradient descent.
        if method == "batch":
            self._batch_gradient_descent(data, label, X_valid, y_valid, learning_rate=learning_rate, epochs=epochs, tol=tol)

        # Stochastic gradient descent.
        if method == "stochastic":
            self._stochastic_gradient_descent(data, label, X_valid, y_valid,
                                              learning_rate=learning_rate, epochs=epochs,
                                              sample_rate=sample_rate, random_state=random_state, tol=tol)



    def predict_proba(self, data: ndarray):

        return self.sigmoid(data.dot(self.weights) + self.bias)

    def predict(self, data: ndarray, threshold=0.5):

        prob = self.predict_proba(data)
        return (prob >= threshold).astype(int)



if __name__ == "__main__":
    def batch():
        print("Tesing the performance of LogisticRegression(batch)...")
        # Train model
        clf = MyLogisticRegression()
        clf.fit(data=data_train, label=label_train, learning_rate=0.1, epochs=1000)
        # Model evaluation
        model_evaluation(clf, data_test, label_test)
        print(clf)


    def stochastic():
        print("Tesing the performance of LogisticRegression(stochastic)...")
        # Train model
        clf = MyLogisticRegression()
        clf.fit(data=data_train, label=label_train, learning_rate=0.01, epochs=100,
                method="stochastic", sample_rate=0.8)
        # Model evaluation
        model_evaluation(clf, data_test, label_test)
        print(clf)
