import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import dill
import numpy as np

from src.train_fixed import checkpoint_load, main


class CheckpointWorkflowTests(unittest.TestCase):
    def test_train_calibrate_and_test_preserve_saved_architecture_and_threshold(self):
        workspace_tests = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=workspace_tests) as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(root.parent, workspace_tests)  # Cleanup stays inside this workspace.
            data = root / "data"
            data.mkdir()
            vocabulary = {key: SimpleNamespace(idx2word={i: str(i) for i in range(size)})
                          for key, size in [("diag_voc", 4), ("pro_voc", 3), ("med_voc", 3)]}
            records = [[[[i % 4, (i + 1) % 4], [0, 1], [0, i % 2 + 1], 0, 0],
                        [[(i + 2) % 4, (i + 3) % 4], [1, 2], [1, 2], 0, 0]] for i in range(6)]
            for filename, value in [("voc_final.pkl", vocabulary), ("records_final.pkl", records),
                                    ("ddi_A_final.pkl", np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))]:
                with (data / filename).open("wb") as stream:
                    dill.dump(value, stream)
            run = root / "run"
            common = ["--device", "cpu", "--data-dir", str(data), "--smoke-test", "--threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(common + ["--architecture", "v2", "--no-molecular", "--dim", "16",
                               "--threshold", "0.575", "--run-dir", str(run)])
                report = main(common + ["--validate", "--resume-path", str(run / "best.pt"),
                                        "--tune-threshold", "--threshold-grid", "0.4", "0.6"])
                self.assertIn(0.575, [item["threshold"] for item in report["candidates"]])
                original = next(item for item in report["candidates"] if item["threshold"] == 0.575)
                self.assertGreaterEqual(report["selected"]["jaccard"], original["jaccard"])
                calibrated = checkpoint_load(run / "calibrated.pt")
                result = main(common + ["--eval", "--resume-path", str(run / "calibrated.pt"),
                                        "--architecture", "v2", "--threshold", "0.99"])
            self.assertEqual(calibrated["architecture"], "v2")
            self.assertEqual(result["metrics"]["threshold"], calibrated["threshold"])
            self.assertFalse(json.loads((run / "best_validation.json").read_text())["target_reached"])
            self.assertTrue((run / "validation_predictions.npz").exists())


if __name__ == "__main__":
    unittest.main()
