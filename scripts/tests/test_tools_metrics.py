
import unittest
import numpy as np
from tools.metrics import accuracy_from_proba, macro_f1_from_proba

class TestToolsMetrics(unittest.TestCase):
    def test_accuracy(self):
        y_true = np.array([0, 1, 2])
        # Perfect prediction
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        acc = accuracy_from_proba(y_true, y_proba)
        self.assertEqual(acc, 1.0)

    def test_f1(self):
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        f1 = macro_f1_from_proba(y_true, y_proba)
        self.assertEqual(f1, 1.0)

if __name__ == '__main__':
    unittest.main()
