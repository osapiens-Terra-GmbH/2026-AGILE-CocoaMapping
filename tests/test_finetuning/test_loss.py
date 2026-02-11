import unittest

import torch
from cocoa_mapping.finetuning.loss import MeanOverSamplesCE

IGNORE_INDEX = 3
"""The ignore index for the loss."""


class MeanOverSamplesCETests(unittest.TestCase):
    """Ensure MeanOverSamplesCE computes per-sample cross entropy correctly."""

    def test_matches_manual_computation_with_ignore_index(self):
        """Test that the function matches the manual computation with ignore index."""
        logits = torch.tensor(
            [
                [
                    [[2.0, -1.0], [0.5, 1.2]],
                    [[-0.5, 1.5], [1.0, -0.5]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                [
                    [[1.2, 0.3], [-2.0, 0.5]],
                    [[-0.7, 2.1], [0.3, -1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
            ]
        )
        masks = torch.tensor(
            [
                [[0, 1], [3, 2]],
                [[2, 3], [3, 1]],
            ]
        )

        loss_module = MeanOverSamplesCE(ignore_index=IGNORE_INDEX)
        computed = loss_module(logits, masks)

        # We will now compute the loss manually for each sample.
        per_sample_losses = []
        for sample_idx in range(logits.shape[0]):  # Iterate over the samples.
            sample_loss = 0.0
            valid_pixels = 0

            # Iterate over the pixels in the sample.
            for row in range(logits.shape[2]):
                for col in range(logits.shape[3]):
                    # Get the target class or skip if it is the ignore index.
                    pixel_class = masks[sample_idx, row, col].item()
                    if pixel_class == IGNORE_INDEX:
                        continue

                    # Get the logits for the pixel.
                    pixel_logits = logits[sample_idx, :, row, col]

                    # Stable log-softmax: subtract max logit, exponentiate, then add it back after summing.
                    max_logit = torch.max(pixel_logits)
                    exp_shifted = torch.exp(pixel_logits - max_logit)
                    log_sum_exp = torch.log(exp_shifted.sum()) + max_logit

                    # Pixel-level cross-entropy = log(sum(exp(logits))) - logit_of_target.
                    pixel_loss = (log_sum_exp - pixel_logits[pixel_class]).item()

                    # Add to the sum
                    sample_loss += pixel_loss
                    valid_pixels += 1

            # Add to the list of per-sample losses.
            per_sample_losses.append(sample_loss / max(valid_pixels, 1))

        # Average over manually computed losses.
        expected = sum(per_sample_losses) / len(per_sample_losses)

        # Test that torch got it correctly.
        self.assertAlmostEqual(computed.item(), expected, places=6)

    def test_samples_with_only_ignore_index_contribute_zero(self):
        """Test that samples with only ignore index contribute zero."""
        logits = torch.randn(1, 3, 2, 2)
        masks = torch.full((1, 2, 2), 3, dtype=torch.long)

        loss_module = MeanOverSamplesCE(ignore_index=3)
        computed = loss_module(logits, masks)

        self.assertEqual(computed.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
