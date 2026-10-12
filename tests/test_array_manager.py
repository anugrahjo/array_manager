"""Basic public API tests; run with python -m unittest discover -s tests -v."""
import unittest

import numpy as np

from array_manager.api import (
    Vector, VectorComponentsDict,
)


class VectorTests(unittest.TestCase):
    def test_initial_values_and_named_views(self):
        components = VectorComponentsDict()
        components["position"] = {"shape": (2,), "vals": np.array([1., 2.])}
        components["velocity"] = {"shape": (2,)}
        vector = Vector(components)
        vector.allocate(setup_views=True)

        np.testing.assert_array_equal(vector.data, [1., 2., 0., 0.])
        vector["velocity"] = [3., 4.]
        np.testing.assert_array_equal(vector.data, [1., 2., 3., 4.])
        vector.data[0] = 5.
        np.testing.assert_array_equal(vector["position"], [5., 2.])
        self.assertEqual(len(vector), 4)

    def test_arithmetic_does_not_modify_original(self):
        components = VectorComponentsDict()
        components["x"] = {"shape": (2,), "vals": np.array([1., 2.])}
        vector = Vector(components)
        vector.allocate(setup_views=True)

        result = (vector + 2.) * 3.
        np.testing.assert_array_equal(result["x"], [9., 12.])
        np.testing.assert_array_equal(vector.data, [1., 2.])
        vector += np.array([3., 4.])
        np.testing.assert_array_equal(vector["x"], [4., 6.])


if __name__ == "__main__":
    unittest.main()
