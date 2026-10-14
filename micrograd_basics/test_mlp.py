"""Standard-library tests: python -m unittest discover -s . -p 'test_*.py'"""
import random
import unittest

from mlp import MLP
from value import Value


class ScalarTests(unittest.TestCase):
    def test_shared_node_gradient(self):
        x = Value(3)
        y = x * x + x
        y.backward()
        self.assertAlmostEqual(x.grad, 7)

    def test_finite_difference(self):
        x = Value(-2)
        y = (x * 4 + 9).relu() ** 2
        y.backward()
        eps = 1e-5
        f = lambda a: max(0, 4 * a + 9) ** 2
        estimate = (f(-2 + eps) - f(-2 - eps)) / (2 * eps)
        self.assertAlmostEqual(x.grad, estimate, places=4)

    def test_mlp_shape_and_gradient(self):
        random.seed(1)
        net = MLP(2, [3, 1])
        output = net([1.0, 2.0])
        self.assertEqual(len(output), 1)
        self.assertEqual(len(net.parameters()), 13)
        output[0].backward()
        self.assertTrue(any(p.grad != 0 for p in net.parameters()))


if __name__ == "__main__":
    unittest.main()
