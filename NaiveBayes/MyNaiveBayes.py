from collections import Counter

import numpy as np
from numpy import ndarray, exp, pi, sqrt


class MyGaussianNB:

    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None

    @staticmethod
    def _get_prior(label: ndarray) -> ndarray:

        cnt = Counter(label)
        prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
        return prior

    def _get_avgs(self, data: ndarray, label: ndarray) -> ndarray:

        return np.array([data[label == i].mean(axis=0)
                         for i in range(self.n_class)])

    def _get_vars(self, data: ndarray, label: ndarray) -> ndarray:

        return np.array([data[label == i].var(axis=0)
                         for i in range(self.n_class)])

    def _get_posterior(self, row: ndarray) -> ndarray:

        return (1 / sqrt(2 * pi * self.vars) * exp(
            -(row - self.avgs)**2 / (2 * self.vars))).prod(axis=1)

    def fit(self, data: ndarray, label: ndarray):

        # Calculate prior probability.
        self.prior = self._get_prior(label)
        # Count number of classes.
        self.n_class = len(self.prior)
        # Calculate the mean.
        self.avgs = self._get_avgs(data, label)
        # Calculate the variance.
        self.vars = self._get_vars(data, label)

        print('prior:', self.prior)
        print('n_class:', len(self.prior))
        print('avgs:', self.avgs)
        print('vars:', self.vars)

    def predict_proba(self, data: ndarray) -> ndarray:

        # Caculate the joint probabilities of each feature and each class.
        likelihood = np.apply_along_axis(self._get_posterior, axis=1, arr=data)

        probs = self.prior * likelihood


        # Scale the probabilities
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    def predict(self, data: ndarray) -> ndarray:

        # Choose the class which has the maximum probability
        return self.predict_proba(data).argmax(axis=1)