
import unittest
import numpy as np
from PIA.augment import PIADirectionalAffineAugmenter

class TestPIAAugment(unittest.TestCase):
    def test_initialization(self):
        aug = PIADirectionalAffineAugmenter(gamma=0.2, n_iters=3)
        self.assertEqual(aug.gamma, 0.2)
        self.assertEqual(aug.n_iters, 3)

    def test_fit_transform(self):
        # Mock data: (Batch, Features)
        X = np.random.rand(10, 310)
        aug = PIADirectionalAffineAugmenter(gamma=0.1, n_iters=2)
        aug.fit(X)
        X_aug = aug.transform(X)
        self.assertEqual(X.shape, X_aug.shape)
        # Check that augmentation actually changed values
        self.assertFalse(np.allclose(X, X_aug), "Augmented data should differ from original")

if __name__ == '__main__':
    unittest.main()
