"""
Module for testing the field from a PMT

"""

import unittest
import numpy as np
from scalarfield import ScalarField
from solvers import LaplacianSolver, LaplacianSolverGPU
from utils import all


pix_per_mm = 10
a=3
b=2
c=4
d=2
e=0.2
f=6

class PMTTestCase(unittest.TestCase):
    """
    Test cases for 

    """

    def test_2d_pmt(self):
        a_p = a * pix_per_mm
        b_p = b * pix_per_mm
        c_p = c * pix_per_mm
        d_p = d * pix_per_mm
        e_p = int(e * pix_per_mm)
        f_p = f * pix_per_mm

        pot=ScalarField(shape=(f_p, a_p+2*c_p+d_p+d_p//2+c_p//2+a_p))

        # pot.solver = LaplacianSolverGPU()

        dynode1_slices = (slice(b_p, b_p+e_p), slice(a_p, a_p+c_p))
        dynode3_slices = (slice(b_p, b_p+e_p), slice(a_p + c_p + d_p, a_p + c_p + d_p + c_p))

        dynode2_slices = (slice(f_p-b_p, f_p-b_p+e_p), slice(a_p+c_p+d_p//2-c_p//2, a_p+c_p+d_p//2+c_p//2))
        dynode4_slices = (slice(f_p-b_p, f_p-b_p+e_p), slice(a_p+2*c_p+d_p+d_p//2-c_p//2, a_p+2*c_p+d_p+d_p//2+c_p//2))

        pot.add_boundary_condition(dynode1_slices, 100)
        pot.add_boundary_condition(dynode2_slices, 200)
        pot.add_boundary_condition(dynode3_slices, 300)
        pot.add_boundary_condition(dynode4_slices, 400)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation()
        pot.apply_conditions()
        pot.show()
    
    def set_pmt_conditions(self, array):
        a=3
        b=2
        c=4
        d=2
        e=0.2
        f=6

        f_p, w = array.shape
        pix_per_mm = f_p // f

        a_p = a * pix_per_mm
        b_p = b * pix_per_mm
        c_p = c * pix_per_mm
        d_p = d * pix_per_mm
        e_p = int(e * pix_per_mm)

        dynode1_slices = (slice(b_p, b_p+e_p), slice(a_p, a_p+c_p))
        dynode3_slices = (slice(b_p, b_p+e_p), slice(a_p + c_p + d_p, a_p + c_p + d_p + c_p))

        dynode2_slices = (slice(f_p-b_p, f_p-b_p+e_p), slice(a_p+c_p+d_p//2-c_p//2, a_p+c_p+d_p//2+c_p//2))
        dynode4_slices = (slice(f_p-b_p, f_p-b_p+e_p), slice(a_p+2*c_p+d_p+d_p//2-c_p//2, a_p+2*c_p+d_p+d_p//2+c_p//2))

        array[*dynode1_slices] = 100
        array[*dynode2_slices] = 200
        array[*dynode3_slices] = 300
        array[*dynode4_slices] = 400

    def test_2d_pmt_with_refinements(self):
        pot=ScalarField(shape=(300, 5*190))

        pot.add_boundary_function(self.set_pmt_conditions)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation_with_refinements(factors=[10,3])
        pot.apply_conditions()
        pot.show(block=True)



if __name__ == "__main__":
    unittest.main()
