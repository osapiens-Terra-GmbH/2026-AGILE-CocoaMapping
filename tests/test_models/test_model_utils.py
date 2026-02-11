import unittest

import torch.nn as nn

from cocoa_mapping.models.model_utils import model_contains_unfrozen_encoder


class ModelContainsUnfrozenEncoderTests(unittest.TestCase):
    """Confirm encoder freezing detection behaves as expected."""

    def test_returns_false_when_model_has_no_encoder(self):
        """Test that the function returns False when the model has no encoder."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 2)

        model = SimpleModel()
        self.assertFalse(model_contains_unfrozen_encoder(model))

    def test_returns_true_when_encoder_has_trainable_params(self):
        """Test that the function returns True when the encoder has trainable parameters."""
        class ModelWithEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(4, 2)
                self.head = nn.Linear(2, 1)

        model = ModelWithEncoder()
        self.assertTrue(model_contains_unfrozen_encoder(model))

    def test_returns_false_when_encoder_is_frozen(self):
        """Test that the function returns False when the encoder is frozen."""
        class FrozenEncoderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(4, 2)
                self.head = nn.Linear(2, 1)
                for param in self.encoder.parameters():
                    param.requires_grad = False

        model = FrozenEncoderModel()
        self.assertFalse(model_contains_unfrozen_encoder(model))


if __name__ == "__main__":
    unittest.main()
