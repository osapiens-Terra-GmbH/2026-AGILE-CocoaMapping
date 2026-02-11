import unittest

from cocoa_mapping.finetuning.combined_dataset import CombinedFinetunePretrainDataset


class _DummyFinetuneDataset:
    """Simple in-memory dataset that mimics finetune samples."""

    def __init__(self, length: int):
        """Initialize the dummy finetune dataset with a given length."""
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return {
            "image": f"finetune_image_{idx}",
            "mask": f"finetune_mask_{idx}",
            "source": "finetune",
        }


class _DummyPretrainDataset:
    """Simple in-memory dataset that mimics pretraining samples."""

    def __init__(self, num_samples: int):
        """Initialize the dummy pretrain dataset with a given number of samples."""
        self.data = [
            (f"pretrain_image_{idx}", f"pretrain_label_{idx}") for idx in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class CombinedDatasetTests(unittest.TestCase):
    """Verify that CombinedFinetunePretrainDataset mixes finetune and pretrain samples correctly."""

    def test_length_and_item_types(self):
        """Test that the dataset has the correct length and item types."""
        dataset = CombinedFinetunePretrainDataset(
            finetune_dataset=_DummyFinetuneDataset(length=4),
            pretrain_dataset=_DummyPretrainDataset(num_samples=10),
            pretrain_ratio_pct=20,
            seed=123,
            shuffle=False,
        )

        # 4 finetune samples + round(4 / 80 * 20) = 1 pretrain sample
        self.assertEqual(len(dataset), 5)

        # Check that the finetune samples are in expected order when not shuffling.
        # We start with 4 finetune samples...
        for idx in range(4):
            sample = dataset[idx]
            self.assertEqual(sample["image"], f"finetune_image_{idx}")
            self.assertEqual(sample["mask"], f"finetune_mask_{idx}")
            self.assertEqual(sample["source"], "finetune")

        # and we now have 1 pretrain sample (again, without shuffling).
        pretrain_sample = dataset[4]
        self.assertTrue(pretrain_sample["image"].startswith("pretrain_image_"))
        self.assertTrue(pretrain_sample["mask"].startswith("pretrain_label_"))
        self.assertNotIn("source", pretrain_sample)

    def test_pretrain_indices_cycle_when_exhausted(self):
        """Test that the pretrain indices get reset when the pretrain dataset is exhausted."""
        dataset = CombinedFinetunePretrainDataset(
            finetune_dataset=_DummyFinetuneDataset(length=5),
            pretrain_dataset=_DummyPretrainDataset(num_samples=3),
            pretrain_ratio_pct=60,
            seed=777,
            shuffle=False,
        )

        # We have 5 finetune samples and 3 pretrain samples.
        finetune_length = dataset.total_num_finetune_samples
        self.assertEqual(finetune_length, 5)
        self.assertEqual(dataset.num_pretrain_samples_target, 8)

        # We should see all 3 pretrain samples when we iterate over the dataset.
        seen_images = []
        for offset in range(len(dataset.pretrain_dataset)):
            pretrain_sample = dataset[finetune_length + offset]
            seen_images.append(pretrain_sample["image"])

        self.assertEqual(len(set(seen_images)), len(dataset.pretrain_dataset))
        self.assertEqual(len(dataset.remaining_pretrain_indices), 0)

        # Next access should trigger a reshuffle and repopulate the pool.
        next_sample = dataset[finetune_length + len(dataset.pretrain_dataset)]
        self.assertTrue(next_sample["image"].startswith("pretrain_image_"))
        self.assertEqual(
            len(dataset.remaining_pretrain_indices),
            len(dataset.pretrain_dataset) - 1,
        )


if __name__ == "__main__":
    unittest.main()
