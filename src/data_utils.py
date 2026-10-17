"""Data preparation and patient-macro metrics for the corrected experiment."""
import hashlib
from pathlib import Path

import dill
import numpy as np
import torch
from sklearn.metrics import average_precision_score


def load_pickle(path):
    with open(path, "rb") as stream:
        return dill.load(stream)


def split_patients(records):
    # Exactly the original patient order and boundaries. Return train, validation, test.
    split = int(len(records) * 2 / 3)
    test_size = (len(records) - split) // 2
    return records[:split], records[split + test_size:], records[split:split + test_size]


def build_ehr_graph(training_records, medication_count):
    adjacency = np.zeros((medication_count, medication_count), dtype=np.float32)
    for patient in training_records:
        for visit in patient:
            medications = np.unique(visit[2])
            adjacency[np.ix_(medications, medications)] = 1
    np.fill_diagonal(adjacency, 0)
    return adjacency


def load_data(data_dir):
    data_dir = Path(data_dir)
    records = load_pickle(data_dir / "records_final.pkl")
    vocabulary = load_pickle(data_dir / "voc_final.pkl")
    sizes = tuple(len(vocabulary[key].idx2word) for key in ("diag_voc", "pro_voc", "med_voc"))
    for patient in records:
        if not patient:
            raise ValueError("Encountered an empty patient record.")
        for visit in patient:
            if len(visit) < 3:
                raise ValueError("Expected at least diagnoses, procedures and medications per visit.")
            for field, size in zip(visit, sizes):
                if any(code < 0 or code >= size for code in field):
                    raise ValueError("Record contains a code outside its vocabulary.")
    ddi = np.asarray(load_pickle(data_dir / "ddi_A_final.pkl"), dtype=np.float32)
    if ddi.shape != (sizes[2], sizes[2]):
        raise ValueError("DDI shape does not match the medication vocabulary.")
    ddi = ((ddi > 0) | (ddi.T > 0)).astype(np.float32)
    np.fill_diagonal(ddi, 0)
    return records, vocabulary, sizes, ddi


def data_fingerprint(data_dir, molecular):
    names = ["records_final.pkl", "voc_final.pkl", "ddi_A_final.pkl"]
    if molecular:
        names += ["ddi_mask_H.pkl", "idx2SMILES.pkl", "substructure_smiles.pkl"]
    return {name: hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest() for name in names}


def prepare_molecules(data_dir, medication_vocabulary, device):
    from rdkit import Chem
    from gnn.utils import graph_batch_from_smile

    data_dir = Path(data_dir)
    molecule_map = load_pickle(data_dir / "idx2SMILES.pkl")
    substructures = load_pickle(data_dir / "substructure_smiles.pkl")
    mask = np.asarray(load_pickle(data_dir / "ddi_mask_H.pkl"))
    if mask.shape != (len(medication_vocabulary.idx2word), len(substructures)):
        raise ValueError("Drug/substructure mask dimensions do not match the vocabularies.")
    smiles, counts = [], []
    for index in range(len(medication_vocabulary.idx2word)):
        code = medication_vocabulary.idx2word[index]
        # Stored values are sets; sort to make graph ordering deterministic across processes.
        candidates = sorted(molecule_map.get(code, []))
        valid = [value for value in candidates if Chem.MolFromSmiles(value) is not None]
        if not valid:
            raise ValueError("No valid molecular SMILES for medication {} ({}).".format(index, code))
        smiles.extend(valid)
        counts.append(len(valid))
    for index, value in enumerate(substructures):
        if Chem.MolFromSmiles(value) is None:
            raise ValueError("Invalid substructure SMILES at index {}.".format(index))
    projection = torch.zeros(len(counts), len(smiles), dtype=torch.float32)
    offset = 0
    for index, count in enumerate(counts):
        projection[index, offset:offset + count] = 1 / count
        offset += count
    graphs = {
        "molecule_graphs": graph_batch_from_smile(smiles).to(device),
        "average_projection": projection.to(device),
        "substructure_graphs": graph_batch_from_smile(substructures).to(device),
    }
    print("Molecular graphs: {} molecules, {} substructures".format(len(smiles), len(substructures)), flush=True)
    return mask, graphs


def patient_average_precision(predictions):
    return [float(np.mean([average_precision_score(y, p) if np.any(y) else 0.0
                           for y, p in zip(truth, probability)]))
            for truth, probability in predictions]


def score_predictions(predictions, ddi, threshold, average_precision=None):
    """Equal weight per patient; equal weight per visit within each patient.

    DDI is the global interacting-pair fraction, matching the original script.
    Each item is (visit-by-drug ground truth, visit-by-drug probabilities).
    """
    if not predictions:
        raise ValueError("Cannot evaluate an empty split.")
    patient_metrics = []
    first_visit, followup = [], []
    actual_medications = 0
    interaction_count = pair_count = medication_count = visit_count = 0
    if average_precision is None:
        average_precision = patient_average_precision(predictions)
    for patient_index, (truth, probability) in enumerate(predictions):
        predicted = probability >= threshold
        truth = truth.astype(bool)
        intersection = (truth & predicted).sum(axis=1)
        union = (truth | predicted).sum(axis=1)
        actual = truth.sum(axis=1)
        recommended = predicted.sum(axis=1)
        precision = intersection / np.maximum(recommended, 1)
        recall = intersection / np.maximum(actual, 1)
        f1 = 2 * intersection / np.maximum(actual + recommended, 1)
        jaccard = intersection / np.maximum(union, 1)
        first_visit.append(float(jaccard[0]))
        if len(jaccard) > 1:
            followup.append(float(np.mean(jaccard[1:])))
        actual_medications += int(actual.sum())
        patient_metrics.append([np.mean(jaccard), average_precision[patient_index],
                                np.mean(precision), np.mean(recall), np.mean(f1)])
        for row in predicted:
            chosen = np.flatnonzero(row)
            medication_count += len(chosen)
            visit_count += 1
            pair_count += len(chosen) * (len(chosen) - 1) // 2
            interaction_count += int(np.triu(ddi[np.ix_(chosen, chosen)], k=1).sum())
    values = np.mean(patient_metrics, axis=0)
    result = dict(zip(("jaccard", "prauc", "precision", "recall", "f1"), map(float, values)))
    result.update(ddi_rate=interaction_count / max(pair_count, 1),
                  avg_medications=medication_count / max(visit_count, 1), threshold=float(threshold),
                  avg_true_medications=actual_medications / max(visit_count, 1),
                  first_visit_jaccard=float(np.mean(first_visit)),
                  followup_jaccard=float(np.mean(followup)) if followup else None,
                  patients=len(predictions), visits=visit_count)
    return result
