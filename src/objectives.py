"""Loss functions and model initialization using training labels only."""
import numpy as np
import torch


def soft_jaccard_loss(logits, target):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=-1)
    union = probability.sum(dim=-1) + target.sum(dim=-1) - intersection
    return (1 - (intersection + 1e-6) / (union + 1e-6)).mean()


def medication_prevalence(training, medication_count, patient_weighting="visit"):
    counts = np.zeros(medication_count, dtype=np.float64)
    total = 0.0
    for patient in training:
        weight = 1 / len(patient) if patient_weighting == "patient" else 1.0
        for visit in patient:
            counts[np.unique(visit[2])] += weight
            total += weight
    return ((counts + 0.5) / (total + 1)).astype(np.float32)


def visit_loss_weight(patient, all_patients, total_visits, weighting):
    # Mean weight across all visits is one. Patient-macro evaluation then matches
    # the expected weighted training objective (optimizer updates remain per visit).
    return total_visits / (all_patients * len(patient)) if weighting == "patient" else 1.0
