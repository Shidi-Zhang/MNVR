import copy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from src.data_utils import build_ehr_graph, score_predictions, split_patients
from src.mnvr_components import AdjAttenAgger
from src.mnvr_plus import MNVRPlus
from src.train_fixed import gradient_report


class DataTests(unittest.TestCase):
    def test_split_matches_original_and_graph_uses_only_train(self):
        records = [[[[], [], [index % 4, (index + 1) % 4]]] for index in range(12)]
        train, validation, test = split_patients(records)
        self.assertEqual(train, records[:8])
        self.assertEqual(test, records[8:10])
        self.assertEqual(validation, records[10:])
        graph = build_ehr_graph([[[[], [], [0, 1]]]], 4)
        self.assertEqual(graph[0, 1], 1)
        self.assertEqual(graph[1, 0], 1)
        self.assertEqual(graph[2, 3], 0)
        self.assertEqual(np.trace(graph), 0)

    def test_metrics_preserve_patient_macro_and_global_ddi(self):
        predictions = [
            (np.array([[1, 0, 0]]), np.array([[0.9, 0.1, 0.1]])),
            (np.array([[1, 0, 0], [1, 0, 0]]), np.array([[0.1, 0.9, 0.9], [0.1, 0.9, 0.9]])),
        ]
        ddi = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        metrics = score_predictions(predictions, ddi, 0.5)
        self.assertAlmostEqual(metrics["jaccard"], 0.5)  # not visit-macro 1/3
        self.assertAlmostEqual(metrics["ddi_rate"], 1.0)
        self.assertAlmostEqual(metrics["avg_medications"], 5 / 3)
        empty = score_predictions(predictions, ddi, 0.99)
        self.assertEqual(empty["ddi_rate"], 0)
        self.assertEqual(empty["jaccard"], 0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gnn.utils import graph_batch_from_smile
        torch.set_num_threads(2)
        cls.graphs = {
            "molecule_graphs": graph_batch_from_smile(["CCO", "CCN", "CCC", "CC(=O)O"]),
            "average_projection": torch.eye(4),
            "substructure_graphs": graph_batch_from_smile(["CC", "CO", "CN"]),
        }

    def setUp(self):
        torch.manual_seed(3407)
        self.adjacency = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
        self.mask = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0]])
        self.model = MNVRPlus((8, 7, 4), self.adjacency, self.adjacency, self.mask,
                          dim=16, patient_dropout=0, graph_dropout=0, gnn_layers=2)
        self.visits = [[[0, 1], [0, 2], [0, 1]], [[3], [1, 4], [2, 3]]]

    def test_active_molecular_branches_receive_gradients(self):
        self.model.train()
        logits = self.model(self.visits, **self.graphs)
        self.assertEqual(tuple(logits.shape), (1, 4))
        F.binary_cross_entropy_with_logits(logits, torch.tensor([[0., 0., 1., 1.]])).backward()
        report = gradient_report(self.model)
        self.assertGreater(report["global_encoder"], 0)
        self.assertGreater(report["substruct_encoder"], 0)
        self.assertGreater(report["substruct_rela"], 0)

    @torch.no_grad()
    def test_current_target_not_input_but_past_medications_are(self):
        self.model.eval()
        static = self.model.encode_static(**self.graphs)
        baseline = self.model(self.visits, static=static)
        changed = copy.deepcopy(self.visits)
        changed[-1][2] = [0]
        torch.testing.assert_close(baseline, self.model(changed, static=static), rtol=0, atol=0)
        changed[0][2] = [2, 3]
        self.assertGreater((baseline - self.model(changed, static=static)).abs().max().item(), 1e-7)
        self.model.use_history = False
        torch.testing.assert_close(self.model(self.visits, static=static), self.model(changed, static=static))

    @torch.no_grad()
    def test_molecular_input_changes_prediction_and_cache_is_exact(self):
        self.model.eval()
        static = self.model.encode_static(**self.graphs)
        baseline = self.model(self.visits, **self.graphs)
        torch.testing.assert_close(baseline, self.model(self.visits, static=static))
        changed = dict(static)
        changed["global"] = torch.randn_like(static["global"])
        self.assertGreater((baseline - self.model(self.visits, static=changed)).abs().max().item(), 1e-6)

    def test_no_training_cache_and_dynamic_dimensions(self):
        with self.assertRaises(ValueError):
            self.model(self.visits, static={})
        baseline = MNVRPlus((8, 7, 4), self.adjacency, self.adjacency, None,
                        dim=32, use_molecular=False, ddi_memory=True)
        output = baseline([[[], [], [0]]])
        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertTrue(torch.isfinite(output).all())
        (output.square().mean() + baseline.ddi_penalty(output)).backward()
        self.assertIsNotNone(baseline.ddi_gcn.node_features.grad)

    def test_fully_masked_drug_has_zero_local_representation(self):
        module = AdjAttenAgger(16)
        output = module(torch.randn(4, 16), torch.randn(3, 16), torch.ones(3), torch.zeros(4, 3, dtype=torch.bool))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output.abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
