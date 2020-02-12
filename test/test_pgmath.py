import unittest
from unittest import TestCase

from src.pgmath import DataRepresentation, Distances
import numpy as np
import math as mt

U = np.array([0, 1, 1, 2]).reshape((2, 2))
V = np.array([1, 3, 1, 0]).reshape((2, 2))

X = DataRepresentation(U, V)


class TestPgMath(TestCase):

    def test_difference_matrix(self):
        expected = [1,2,0,1,1,-1,0,-2]
        expected = np.array(expected).reshape((2,2,2))
        d_mat = Distances.difference_matrix(X)
        self.assertTrue(np.allclose(d_mat, expected))

    def test_delta2(self):
        expected = np.array([mt.sqrt(5), 1, mt.sqrt(2), 2]).reshape((2, 2))
        d2 = Distances.delta2(X)
        self.assertTrue(np.allclose(d2, expected))

    def test_zeta2(self):
        expected = np.array([mt.exp(-5) + mt.exp(-1), mt.exp(-2) + mt.exp(-4)])
        z2 = Distances.zeta2(Distances.delta2(X))
        self.assertTrue(np.allclose(z2, expected))


if __name__ == '__main__':
    unittest.main()
