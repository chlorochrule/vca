#-*- coding: utf-8 -*-
""" Vanishing Component Analysis
"""

# Author: Naoto MINAMI <minami@cmu.iit.tsukuba.ac.jp>
#
# License: BSD 3 clause

import numpy as np
from numpy.linalg import svd, norm
from sympy import Symbol, plot_implicit

class VCA(object):
    """Vanishing component analysis
    """
    def __init__(self, eps=0.05, n_component=None, max_dimension=3):
        super(VCA, self).__init__()
        self.eps = eps
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
        m, n = S.shape
        F = [1./np.sqrt(m)]
        V = []
        FS = np.array([[1./np.sqrt(m)]*m])
        C = [Symbol('x{}'.format(i+1)) for i in xrange(n)]
        CS = S.T
        F1, FS1, V1 = self._find_range_null(F, FS, C, CS)
        F.extend(F1)
        FS = np.append(FS, FS1, axis=0)
        V.extend(V1)
        Ft = F1
        FSt = FS1
        for i in xrange(1, self.max_dimension+1):
            C = [g * h for g in Ft for h in F1]
            CS = np.repeat(FSt, len(FS1), axis=0) * np.tile(FS1, (len(FSt), 1))
            if len(C) == 0:
                break
            Ft, FSt, Vt = self._find_range_null(F, FS, C, CS)
            F.extend(Ft)
            FS = np.append(FS, FSt, axis=0)
            V.extend(Vt)
        self.components_ = V

    def _find_range_null(self, F, FS, C, CS):
        k = len(C)
        inn_FCS = np.dot(CS, FS.T)
        for i, f in enumerate(C):
            for j, g in enumerate(F):
                C[i] -= inn_FCS[i][j] * g
        A = (CS - np.dot(inn_FCS, FS)).T
        _, D, U = svd(A)
        GS = np.dot(U, CS)
        F1 = []
        V1 = []
        dmax = len(D)
        for i in xrange(k):
            g = 0
            for j, fc in enumerate(C):
                g += U[i][j] * fc
            if i >= dmax:
                D = np.append(D, 0)
            if D[i] > self.eps:
                F1.append(g / norm(GS[i]))
            else:
                V1.append(g)
        normalize = lambda x: x / norm(x)
        FS1 = np.apply_along_axis(normalize, 1, GS[D > self.eps]) if max(D) > self.eps else GS[D > self.eps]
        return F1, FS1, V1

def main():
    vca = VCA(eps=0.005, max_dimension=4)
    vca.fit(np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]))
    print vca.components_
    for fn in vca.components_:
        plot_implicit(fn)


if __name__ == '__main__':
    main()
