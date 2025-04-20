"""
Module for testing the gradient of the ScalarField class and its methods using unittest.


"""

import unittest
import numpy as np
from scalarfield import ScalarField
from utils import all


class GradientTestCase(unittest.TestCase):
    """ """

    def test_init(self):
        """Test that a ScalarField instance is initialized properly."""
        self.assertIsNotNone(ScalarField(shape=(32, 32)))

    @unittest.SkipTest
    def test_linear_gradient(self):
        potential = ScalarField(shape=(32, 64))

        gradient_line = np.linspace(0, 1, potential.shape[-1], dtype=np.float32)
        potential.values = np.broadcast_to(gradient_line, potential.shape)
        potential.show()

    @unittest.SkipTest
    def test_field_is_linear_axis_0(self):
        potential = ScalarField(shape=(32, 64))
        potential.set_linear_gradient(potential.shape, axis=0)
        potential.show()

    @unittest.SkipTest
    def test_field_is_linear_axis_1(self):
        potential = ScalarField(shape=(32, 64))
        potential.set_linear_gradient(potential.shape, axis=1)
        potential.show()

    def test_linear_in_x_field_null_gradient_in_x(self):
        """
        A linear field in x gives a constant gradient.
        Field goes from 0 to 1, gradient will be 0.1 in x
        and 0 in y (axis 0 and 1)
        """

        potential = ScalarField(shape=(10, 10))
        potential.set_linear_gradient(potential.shape, axis=0)
        x_grad, y_grad = potential.gradient()

        for v_line in x_grad[1:-1, :]:
            for v in v_line:
                self.assertAlmostEqual(v, 0.1, 4)

        for v_line in y_grad:
            for v in v_line:
                self.assertAlmostEqual(v, 0.0, 4)

        potential.show()

    def test_linear_in_y_field_null_gradient_in_y(self):
        """
        A linear field in y gives a constant gradient.
        Field goes from 0 to 1, gradient will be 0.1 in y
        and 0 in x (axis 0 and 1)
        """

        potential = ScalarField(shape=(10, 10))
        potential.set_linear_gradient(potential.shape, axis=1)
        x_grad, y_grad = potential.gradient()

        for v_line in x_grad[1:-1, :]:
            for v in v_line:
                self.assertAlmostEqual(v, 0.0, 4)

        for v_line in y_grad:
            for v in v_line:
                self.assertAlmostEqual(v, 0.1, 4)

        potential.show()


if __name__ == "__main__":
    unittest.main()
