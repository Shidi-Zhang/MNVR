import unittest

import numpy as np
import torch
from torch.nn import functional as F

import test_mnvr
from src.mnvr_plus import MNVRPlus
from src.objectives import medication_prevalence, soft_jaccard_loss, visit_loss_weight
from src.train_fixed import apply_defaults, parser, main, warm_start


class PlusModelTests(test_mnvr.ModelTests):
    # Reuse molecular sensitivity, no-target-leakage, caching and gradient checks for V2.
    def setUp(self):
        super().setUp()
        self.model = MNVRPlus((8, 7, 4), self.adjacency, self.adjacency, self.mask,
                              dim=16, patient_dropout=0, graph_dropout=0, gnn_layers=2)

    def test_copy_gate_and_direct_heads_train(self):
        logits = self.model(self.visits, **self.graphs)
        F.binary_cross_entropy_with_logits(logits, torch.tensor([[0., 0., 1., 1.]])).backward()
        self.assertGreater(self.model.copy_gate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.direct_heads[0].weight.grad.abs().sum().item(), 0)

    @torch.no_grad()
    def test_copy_can_be_disabled_or_select_previous_prescription(self):
        self.model.eval()
        static = self.model.encode_static(**self.graphs)
        self.model.use_copy = False
        generator = self.model(self.visits, static=static)
        self.model.use_copy = True
        self.model.copy_gate.weight.zero_()
        self.model.copy_gate.bias.fill_(-100)
        torch.testing.assert_close(generator, self.model(self.visits, static=static))
        self.model.copy_gate.bias.fill_(100)
        probability = torch.sigmoid(self.model(self.visits, static=static))[0]
        self.assertTrue(torch.all(probability[:2] > 0.999))
        self.assertTrue(torch.all(probability[2:] < 1e-5))

    def test_prior_initialization_is_finite_for_rare_drugs(self):
        self.model.initialize_prevalence([0.0, 0.2, 0.8, 1.0])
        self.assertTrue(torch.isfinite(self.model.output[-1].bias).all())
        torch.testing.assert_close(torch.sigmoid(self.model.output[-1].bias)[1:3], torch.tensor([0.2, 0.8]))

    def test_clinical_warm_start_preserves_new_graph_and_adds_molecules(self):
        source = MNVRPlus((8, 7, 4), np.zeros((4, 4)), self.adjacency, None,
                          dim=16, patient_dropout=0, graph_dropout=0, gnn_layers=2, use_molecular=False)
        graph = self.model.ehr_gcn.adjacency.clone()
        checkpoint = {"architecture": "v2", "data_fingerprint": {"records": "same"},
                      "model_config": {"use_molecular": False}, "model_state": source.state_dict()}
        missing = warm_start(self.model, checkpoint, "v2", {"records": "same", "molecules": "extra"})
        self.assertGreater(missing, 0)
        torch.testing.assert_close(self.model.output[-1].weight, source.output[-1].weight)
        torch.testing.assert_close(self.model.ehr_gcn.adjacency, graph)
        with self.assertRaisesRegex(ValueError, "data files"):
            warm_start(self.model, checkpoint, "v2", {"records": "changed"})


class ObjectiveTests(unittest.TestCase):
    def test_jaccard_gradient_rewards_true_drugs_and_discourages_false_drugs(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        target = torch.tensor([[1., 0., 1., 0.]])
        soft_jaccard_loss(logits, target).backward()
        self.assertTrue(torch.all(logits.grad[target.bool()] < 0))
        self.assertTrue(torch.all(logits.grad[~target.bool()] > 0))

    def test_patient_weighting_and_prior_use_only_passed_records(self):
        short = [[[0], [0], [0]]]
        long = [[[0], [0], [1]]] * 3
        self.assertEqual(visit_loss_weight(short, 2, 4, "patient"), 2)
        self.assertAlmostEqual(visit_loss_weight(long, 2, 4, "patient") * 3, 2)
        prior = medication_prevalence([short, long], 3, "patient")
        np.testing.assert_allclose(prior, [0.5, 0.5, 1 / 6])

    def test_v2_training_defaults_and_test_calibration_guard(self):
        args = apply_defaults(parser().parse_args(["--architecture", "v2"]))
        self.assertEqual(args.optimizer, "adamw")
        self.assertEqual(args.margin_weight, 0.0)
        self.assertEqual(args.jaccard_weight, 0.15)
        self.assertEqual(args.patient_weighting, "patient")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            main(["--eval", "--tune-threshold", "--resume-path", "unused.pt"])


if __name__ == "__main__":
    unittest.main()
