#-*- coding: utf-8 -*-
""" Vanishing Component Analysis
"""

# Author: Naoto MINAMI <minami@cmu.iit.tsukuba.ac.jp>
#
# License: BSD 3 clause

import numpy as np
from numpy.linalg import svd, norm

class VCA(object):
    """Vanishing component analysis
    """
    def __init__(self, e=0.05, n_component=None, max_dimension=3):
        super(VCA, self).__init__()
        self.e = e
        self.n_component = n_component
        self.max_dimension = max_dimension

    def fit(self, X):
        """Fit the model with X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def _fit(self, S):
        m = len(X)
        F = np.ndarray([[1/np.sqrt(m)]])
        C = np.matrix(np.identity(m+1))[1:]
        power_matrix = np.matrix(np.identity(m+1))
        F1, V1 = _find_range_null(F, C, S, power_matrix)
        F.extend(F1)
        V.extend(V1)
        Ft = F1
        for i in xrange(1, m):
            power_matrix = self._update_power_matrix(Ft, F1, power_matrix)
            C = self._update_ct(Ft, F1, power_matrix)
            if len(C) == 0:
                break
            Ft, Vt = _find_range_null(F, C, S, power_matrix)
            F.extend(Ft)
            V.extend(Vt)
            F = self._pad(F, power_matrix.shape[1])
            V = self._pad(V, power_matrix.shape[1])

    def _find_range_null(self, F, C, S, power_matrix):
        C = self._gs(C, F, S, power_matrix)
        A = self._evalate(C, S, power_matrix)
        _, D, U = svd(A)
        G = np.dot(U, C)
        Dii = np.sum(D, axis=1)
        normalize = lambda x: x / norm(x)
        return G[Dii > self.e].apply_along_axis(normalize), G[Dii <= self.e]

    def _evalate(self, C, S, power_matrix):
        monomialize = lambda x: (x ** power_matrix.T).prod(axis=1)
        return np.dot(self._pad(C, power_matrix.shape[1]), (S.apply_along_axis(monomialize).T)).T

    def _pad(self, matrix, length):
        pad_func = lambda x: np.pad(x, (0, length - len(x)), 'constant')
        return matrix.apply_along_axis(pad_func)

    def _gs(C, F, S, power_matrix):
        A = self._evalate(C, S, power_matrix).T
        B = self._evalate(F, S, power_matrix)
        C -= np.dot(np.dot(A, B) , F)
        return C

    def _update_power_matrix(Ft, F1, power_matrix):
        
