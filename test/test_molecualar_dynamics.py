# -*- coding: utf-8 -*-
import unittest
import numpy as np
from surface_hopping.molecular_dynamics import update_x
from surface_hopping.molecular_dynamics import update_v
from surface_hopping.molecular_dynamics import call_surface_hopping
from surface_hopping.molecular_dynamics import add_decoherence


class TestMolecularDynamics(unittest.TestCase):
    """Testing functionality of the surface hopping molecular dynamics
    """

    def setUp(self):
        """ set up tolerance"""

        self.tol = 1e-7

    
    def test_update_x(self):
        """ Test velocity verlet position update function"""

        x_test = 1.0
        v_test = 1.0
        a_test = 0.1
        dt_test = 0.2
        x_updated_test = 1.202
        x_updated = update_x(x_test, v_test, a_test, dt_test)

        self.assertTrue(np.isclose(x_updated_test, x_updated, rtol=self.tol))


    def test_update_v(self):
        """ Test velocity verlet momentum update function"""

        v_test = 1.0
        a_curr_test = 0.1
        a_new_test = 0.15
        dt_test = 0.2
        v_updated_test = 1.025
        v_updated = update_v(v_test, a_curr_test, a_new_test, dt_test)

        self.assertTrue(np.isclose(v_updated_test, v_updated, rtol=self.tol))


    def test_call_surface_hopping(self):
        """ Test velocity verlet momentum update function

        
        Notes
        -----
        Surface Hopping is a probabilistic a;gorithm by design. This function 
        generates random numbers and therefore cannot be tested in usual fashion.
        There are some ways to test it but this is for later.

        See: https://www.random.org/analysis/

        """

    
