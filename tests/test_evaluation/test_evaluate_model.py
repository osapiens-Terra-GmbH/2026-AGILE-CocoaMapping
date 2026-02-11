import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import shutil

import cocoa_mapping.evaluation.evaluate_model as evaluate_model


class TestDownloadImages(unittest.TestCase):
    def setUp(self):
        # Temporary directory for tests that need it
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.tmpdir)

    def test_download_images_sentinel_calls_downloader(self):
        """Non-AEF mode should delegate to `download_and_consolidate_tile` with yearly interval."""
        captured = {}

        def fake_download_and_consolidate_tile(**kwargs):
            captured.update(kwargs)
            return ["scene1.tif", "scene2.tif"]

        with patch.object(evaluate_model, "download_and_consolidate_tile", fake_download_and_consolidate_tile):
            result = evaluate_model.download_images(
                grid_code="29NQJ",
                aef_mode=False,
                output_dir="/tmp/out",
                year=2022,
                delete_input=None,
                num_scenes=3,
                mininterval=0.5,
            )

        self.assertEqual(result, ["scene1.tif", "scene2.tif"])
        self.assertEqual(captured["grid_code"], "29NQJ")
        self.assertEqual(captured["time_interval"], "2022-01-01/2022-12-31")
        self.assertEqual(captured["num_scenes"], 3)
        self.assertEqual(captured["output_dir"], "/tmp/out")
        self.assertEqual(captured["tqdm_mininterval"], 0.5)

    def test_download_images_aef_calls_alphaearth(self):
        """AEF mode should call the AlphaEarth downloader with the provided year."""
        captured = {}

        def fake_download_aef_for_sentinel_2_tile(**kwargs):
            captured.update(kwargs)
            return "/tmp/aef_2021.tif"

        output_dir = Path(self.tmpdir) / "aef"
        output_dir.mkdir()

        with patch.object(evaluate_model, "download_aef_for_sentinel_2_tile", fake_download_aef_for_sentinel_2_tile):
            result = evaluate_model.download_images(
                grid_code="29NQJ",
                aef_mode=True,
                output_dir=str(output_dir),
                year=2021,
                delete_input=True,
                num_scenes=None,
                mininterval=0.1,
            )

        self.assertEqual(result, ["/tmp/aef_2021.tif"])
        self.assertEqual(captured["grid_code"], "29NQJ")
        self.assertEqual(captured["year"], 2021)
        self.assertEqual(captured["output_path"], str(output_dir / "aef_2021.tif"))
        self.assertTrue(captured["delete_input"])
        self.assertTrue(captured["use_progress_callback"])
        self.assertEqual(captured["max_download_workers"], 8)


if __name__ == "__main__":
    unittest.main()
