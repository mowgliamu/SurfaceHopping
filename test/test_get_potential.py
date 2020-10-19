# -*- coding: utf-8 -*-
import unittest
import numpy as np
from surface_hopping.get_potential import get_energies
from surface_hopping.get_potential import get_gradients_and_nadvec
from surface_hopping.get_potential import get_gradient_numerical


class TestGetPotential(unittest.TestCase):
    """Testing functionality of the vibronic model potential functions
    """

    def setUp(self):
        """ set up tolerance"""

        self.tol = 1e-7

    def test_get_energies(self):
        """ Test vibronic model energies"""
        
        x_test = 1.0
        e1_test = np.array([-0.00819026,  0.00819026])
        e2_test = np.array([[ 0.11301587, -0.99359318], 
                           [-0.99359318, -0.11301587]])
        e1, e2 = get_energies(x_test)
        
        self.assertTrue(np.isclose(e1, e1_test, rtol=self.tol).all())
        self.assertTrue(np.isclose(e2, e2_test, rtol=self.tol).all())

    def test_get_gradient_and_nadvec(self):
        """ Test vibronic model gradients obtained anaylitically"""

        x_test = 1.0
        en_test = np.array([-0.00819026,  0.00819026])
        grad_test = np.array([[-0.00232163, 0.00232163]])
        nadvec_test = np.array([0.26313592])
        en, grad, nadvec = get_gradients_and_nadvec(x_test)

        self.assertTrue(np.isclose(en, en_test, rtol=self.tol).all())
        self.assertTrue(np.isclose(grad, grad_test, rtol=self.tol).all())
        self.assertTrue(np.isclose(nadvec, nadvec_test, rtol=self.tol).all())

    def test_get_gradient_numerical(self):
        """ Test vibronic model gradients obtained numerically"""

        x_test = 1.0
        grad_test = np.array([[-0.00232163, 0.00232163]])
        grad = get_gradient_numerical(x_test)

        self.assertTrue(np.isclose(grad, grad_test, rtol=self.tol).all())

    
if __name__ == '__main__':
    unittest.main()

