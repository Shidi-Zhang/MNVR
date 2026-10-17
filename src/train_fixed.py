"""CPU/GPU training, validation-only selection, and independent test evaluation."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.data_utils import (build_ehr_graph, data_fingerprint, load_data,
                            prepare_molecules, score_predictions, split_patients, patient_average_precision)
from src.mnvr_plus import MNVRPlus
from src.objectives import medication_prevalence, soft_jaccard_loss, visit_loss_weight

ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--data-dir", type=Path, default=ROOT / "data")
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--model-name", "--model_name", default="MNVR_fixed")
    result.add_argument("--architecture", choices=("v2",), default="v2")
    result.add_argument("--seed", type=int, default=3407)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--lr", type=float, default=0.00015)
    result.add_argument("--weight-decay", type=float, default=None)
    result.add_argument("--optimizer", choices=("adam", "adamw"), default=None)
    result.add_argument("--scheduler", choices=("none", "plateau"), default=None)
    result.add_argument("--lr-patience", type=int, default=3)
    result.add_argument("--dim", type=int, default=64)
    result.add_argument("--patient-dropout", type=float, default=0.3)
    result.add_argument("--graph-dropout", "--dp", type=float, default=0.3)
    result.add_argument("--gnn-layers", type=int, default=4)
    result.add_argument("--bce-weight", type=float, default=None)
    result.add_argument("--margin-weight", type=float, default=None)
    result.add_argument("--jaccard-weight", type=float, default=None)
    result.add_argument("--patient-weighting", choices=("visit", "patient"), default=None)
    result.add_argument("--ddi-weight", type=float, default=0.0)
    result.add_argument("--ddi", action="store_true", help="Enable DDI penalty (weight 0.05 unless explicitly set positive).")
    result.add_argument("--ddi-memory", action="store_true", help="Ablation: subtract a learned DDI graph representation.")
    result.add_argument("--no-molecular", action="store_true")
    result.add_argument("--no-history", action="store_true")
    result.add_argument("--no-copy", action="store_true", help="V2 ablation: disable the per-drug history mixture.")
    result.add_argument("--no-direct", action="store_true", help="V2 ablation: disable direct code-to-drug scores.")
    result.add_argument("--grad-clip", type=float, default=5.0)
    result.add_argument("--patience", type=int, default=15)
    result.add_argument("--threads", type=int, default=4, help="CPU intra-op threads.")
    result.add_argument("--threshold", type=float, default=0.5, help="Fixed training validation threshold; checkpoint evaluation uses its saved threshold.")
    result.add_argument("--tune-threshold", action="store_true", help="Select a global threshold on validation ONLY.")
    result.add_argument("--threshold-grid", type=float, nargs="+", default=[round(0.3 + i * 0.025, 3) for i in range(13)])
    result.add_argument("--target-jaccard", type=float, default=0.54, help="Reporting target only; never changes measured metrics.")
    result.add_argument("--max-ddi", type=float, help="Optional validation DDI constraint for checkpoint selection.")
    result.add_argument("--smoke-test", action="store_true", help="Two patients per split and at most three training steps; NOT a performance run.")
    result.add_argument("--max-train-patients", type=int)
    result.add_argument("--max-eval-patients", type=int)
    result.add_argument("--max-train-steps", type=int)
    result.add_argument("--check-gradients", action="store_true")
    result.add_argument("--log-every", type=int, default=100)
    result.add_argument("--eval", action="store_true", help="Evaluate test split with the checkpoint's fixed threshold.")
    result.add_argument("--validate", action="store_true", help="Inspect/calibrate an existing checkpoint using validation only.")
    result.add_argument("--resume-path", "--resume_path", type=Path, help="Checkpoint for --eval or --validate.")
    result.add_argument("--init-from", type=Path, help="Warm-start a new run, optionally adding the molecular branch. Optimizer starts fresh.")
    return result


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def checkpoint_load(path):
    # Only load checkpoints supplied by the user / produced by this project.
    try:
        result = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        result = torch.load(path, map_location="cpu")
    if not isinstance(result, dict) or (result.get("format") != "mnvr_plus_v2" or result.get("architecture") != "v2"):
        raise ValueError("Only MNVR V2 checkpoints are supported.")
    return result


def warm_start(model, checkpoint, architecture, fingerprint):
    if checkpoint.get("architecture") != architecture:
        raise ValueError("Warm-start requires the same architecture version.")
    if any(fingerprint.get(key) != value for key, value in checkpoint["data_fingerprint"].items()):
        raise ValueError("Warm-start data files differ from the source checkpoint.")
    molecular_prefixes = ("global_encoder.", "substruct_encoder.", "sab.", "substruct_rela.",
                          "aggregator.", "molecular_norm.", "molecular_query.",
                          "molecular_scale", "substructure_mask")
    # A source trained on fewer training patients must not overwrite the graph
    # built for the new run. Its learned weights can still be useful initialization.
    current_graph = model.ehr_gcn.adjacency.detach().clone()
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.ehr_gcn.adjacency.copy_(current_graph)
    adding_molecules = model.use_molecular and not checkpoint["model_config"]["use_molecular"]
    if incompatible.unexpected_keys or any(
            not adding_molecules or not name.startswith(molecular_prefixes)
            for name in incompatible.missing_keys):
        raise ValueError("Warm-start configuration mismatch: {}".format(incompatible))
    return len(incompatible.missing_keys)


@torch.no_grad()
def predict(model, patients, graph_inputs):
    model.eval()
    static = model.encode_static(**graph_inputs)
    predictions = []
    for patient in patients:
        truths, probabilities = [], []
        for index, visit in enumerate(patient):
            logits = model(patient[:index + 1], static=static)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits.")
            truth = np.zeros(model.vocab_size[2], dtype=np.float32)
            truth[visit[2]] = 1
            truths.append(truth)
            probabilities.append(torch.sigmoid(logits).cpu().numpy()[0])
        predictions.append((np.stack(truths), np.stack(probabilities)))
    return predictions


def gradient_report(model):
    names = ["embeddings", "encoders", "ehr_gcn", "output"]
    if model.use_molecular:
        names += ["global_encoder", "substruct_encoder", "sab", "substruct_rela", "aggregator", "molecular_query"]
    if model.ddi_memory:
        names += ["ddi_gcn"]
    if isinstance(model, MNVRPlus):
        names += ["code_pooling", "current_projection"]
        if model.use_direct:
            names += ["direct_heads"]
    report = {}
    for name in names:
        parameters = list(getattr(model, name).parameters())
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Missing/non-finite gradient in {}".format(name))
        magnitude = math.sqrt(sum(float(gradient.detach().square().sum()) for gradient in gradients))
        if magnitude == 0:
            raise RuntimeError("Zero gradient in active branch {}".format(name))
        report[name] = magnitude
    return report


def train_epoch(model, patients, graphs, optimizer, args, epoch):
    model.train()
    device = next(model.parameters()).device
    totals = np.zeros(5, dtype=float)
    count = 0
    total_visits = sum(len(patient) for patient in patients)
    order = np.random.permutation(len(patients))
    for patient_index in order:
        patient = patients[int(patient_index)]
        for visit_index, visit in enumerate(patient):
            target = torch.zeros(1, model.vocab_size[2], device=device)
            target[0, visit[2]] = 1
            margin_target = torch.full(target.shape, -1, dtype=torch.long, device=device)
            medications = sorted(set(visit[2]))
            margin_target[0, :len(medications)] = torch.tensor(medications, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(patient[:visit_index + 1], **graphs)
            bce = F.binary_cross_entropy_with_logits(logits, target)
            margin = F.multilabel_margin_loss(torch.sigmoid(logits), margin_target)
            ddi = model.ddi_penalty(logits)
            jaccard = soft_jaccard_loss(logits, target)
            loss = (args.bce_weight * bce + args.margin_weight * margin +
                    args.ddi_weight * ddi + args.jaccard_weight * jaccard)
            loss = loss * visit_loss_weight(patient, len(patients), total_visits, args.patient_weighting)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            if epoch == 0 and count == 0 and (args.check_gradients or args.smoke_test):
                print("Gradient norms: " + json.dumps(gradient_report(model)), flush=True)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm.")
            optimizer.step()
            totals += [loss.item(), bce.item(), margin.item(), ddi.item(), jaccard.item()]
            count += 1
            if count % args.log_every == 0:
                print("Epoch {} step {} loss {:.5f}".format(epoch, count, totals[0] / count), flush=True)
            if args.max_train_steps and count >= args.max_train_steps:
                return dict(zip(("total", "bce", "margin", "ddi", "soft_jaccard"), (totals / count).tolist())), count
    return dict(zip(("total", "bce", "margin", "ddi", "soft_jaccard"), (totals / count).tolist())), count


def apply_defaults(args):
    defaults = dict(bce_weight=1.0, margin_weight=0.0, jaccard_weight=0.15,
                    patient_weighting="patient", optimizer="adamw", weight_decay=1e-5, scheduler="plateau")
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def select_validation(predictions, ddi, args):
    average_precision = patient_average_precision(predictions)
    fixed = score_predictions(predictions, ddi, 0.5, average_precision)
    thresholds = sorted(set(args.threshold_grid + [args.threshold])) if args.tune_threshold else [args.threshold]
    candidates = [fixed if threshold == 0.5 else score_predictions(predictions, ddi, threshold, average_precision)
                  for threshold in thresholds]
    eligible = [item for item in candidates if args.max_ddi is None or item["ddi_rate"] <= args.max_ddi]
    selected = max(eligible, key=lambda item: (item["jaccard"], -abs(item["threshold"] - 0.5))) if eligible else None
    return fixed, candidates, selected


def save_predictions(path, predictions):
    offsets = np.cumsum([0] + [len(truth) for truth, _ in predictions])
    np.savez_compressed(path, truth=np.concatenate([truth for truth, _ in predictions]),
                        probability=np.concatenate([probability for _, probability in predictions]),
                        patient_offsets=offsets)


def main(argv=None):
    args = apply_defaults(parser().parse_args(argv))
    if args.epochs < 1 or args.lr <= 0 or args.grad_clip <= 0 or args.threads < 1 or args.log_every < 1:
        raise ValueError("epochs, lr, grad-clip, threads and log-every must be positive.")
    for value in (args.max_train_patients, args.max_eval_patients, args.max_train_steps):
        if value is not None and value < 1:
            raise ValueError("Subset/step limits must be positive.")
    if any(value < 0 for value in (args.bce_weight, args.margin_weight, args.ddi_weight, args.jaccard_weight, args.weight_decay)):
        raise ValueError("Loss weights and weight decay must be nonnegative.")
    if any(not 0 < value < 1 for value in [args.threshold] + args.threshold_grid):
        raise ValueError("Thresholds must lie between zero and one.")
    if args.max_ddi is not None and not 0 <= args.max_ddi <= 1:
        raise ValueError("max-ddi must lie between zero and one.")
    if args.eval and args.tune_threshold:
        raise ValueError("Threshold tuning is forbidden on the test split.")
    if args.eval and args.validate:
        raise ValueError("Choose --eval OR --validate, never both.")
    if (args.eval or args.validate) != (args.resume_path is not None):
        raise ValueError("Use --resume-path with --eval or --validate. Training resume is not implemented.")
    if args.init_from is not None and (args.eval or args.validate):
        raise ValueError("--init-from is only for a new training run.")
    if args.lr_patience < 0 or not 0 < args.target_jaccard <= 1:
        raise ValueError("Invalid lr-patience or target-jaccard.")
    if args.ddi and args.ddi_weight == 0:
        args.ddi_weight = 0.05
    if args.smoke_test:
        args.epochs = 1
        args.max_train_patients = 2
        args.max_eval_patients = 2
        args.max_train_steps = 3
        print("SMOKE TEST ONLY: metrics are not performance estimates.", flush=True)
    elif args.max_train_patients or args.max_eval_patients or args.max_train_steps:
        print("SUBSET RUN: diagnostic experiment, not a full-data benchmark.", flush=True)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use --device cpu.")
    print("Device: {}; PyTorch: {}; NumPy: {}".format(device, torch.__version__, np.__version__), flush=True)
    records, vocabulary, sizes, ddi = load_data(args.data_dir)
    full_train, validation, test = split_patients(records)
    if not full_train or not validation or not test:
        raise ValueError("All three patient splits must be nonempty.")
    print("Patient split: train={}, validation={}, test={}".format(len(full_train), len(validation), len(test)), flush=True)
    checkpoint = checkpoint_load(args.resume_path) if (args.eval or args.validate) else None
    architecture = "v2"
    config = checkpoint["model_config"] if checkpoint else dict(
        vocab_size=sizes, dim=args.dim, patient_dropout=args.patient_dropout,
        graph_dropout=args.graph_dropout, gnn_layers=args.gnn_layers,
        use_molecular=not args.no_molecular, use_history=not args.no_history,
        ddi_memory=args.ddi_memory)
    if not checkpoint:
        config.update(use_copy=not args.no_copy, use_direct=not args.no_direct)
    fingerprint = data_fingerprint(args.data_dir, config["use_molecular"])
    if checkpoint and checkpoint["data_fingerprint"] != fingerprint:
        raise ValueError("Data/vocabulary files differ from the training checkpoint.")
    if checkpoint and checkpoint["smoke_test"] and not args.smoke_test:
        raise ValueError("This is a smoke-test checkpoint; pass --smoke-test for diagnostic evaluation only.")
    graphs, mask = {}, None
    if config["use_molecular"]:
        mask, graphs = prepare_molecules(args.data_dir, vocabulary["med_voc"], device)
    # Use only training patients and exactly the current drug indices. Never use test/validation labels.
    training = full_train[:args.max_train_patients]
    graph_training = full_train if checkpoint else training
    ehr = build_ehr_graph(graph_training, sizes[2])
    model = MNVRPlus(ehr_adj=ehr, ddi_adj=ddi, ddi_mask_H=mask, **config).to(device)
    if not checkpoint:
        model.initialize_prevalence(medication_prevalence(training, sizes[2], args.patient_weighting))
    if args.init_from is not None:
        source = checkpoint_load(args.init_from)
        if source["smoke_test"] and not args.smoke_test:
            raise ValueError("Do not initialize a performance run from a smoke-test checkpoint.")
        missing = warm_start(model, source, architecture, fingerprint)
        print("Warm-start loaded; {} newly initialized molecular tensors. Optimizer starts fresh.".format(missing), flush=True)
    print("Architecture: {}".format(architecture), flush=True)
    print("Parameters: {:,}".format(sum(p.numel() for p in model.parameters())), flush=True)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        if args.validate:
            args.threshold = checkpoint["threshold"]
            predictions = predict(model, validation[:args.max_eval_patients], graphs)
            fixed, candidates, selected = select_validation(predictions, ddi, args)
            report = dict(split="validation", checkpoint=str(args.resume_path.resolve()),
                          fixed_05=fixed, candidates=candidates, selected=selected,
                          smoke_test=checkpoint["smoke_test"])
            write_json(args.resume_path.parent / "validation_metrics.json", report)
            save_predictions(args.resume_path.parent / "validation_predictions.npz", predictions)
            if args.tune_threshold and selected is not None:
                calibrated = dict(checkpoint)
                calibrated.update(threshold=selected["threshold"], validation_metrics=selected,
                                  validation_fixed_05=fixed, calibration_source="validation",
                                  calibration_patients=len(predictions))
                torch.save(calibrated, args.resume_path.parent / "calibrated.pt")
            print(json.dumps(report, indent=2), flush=True)
            return report
        metrics = score_predictions(predict(model, test[:args.max_eval_patients], graphs), ddi, checkpoint["threshold"])
        report = dict(split="test", checkpoint=str(args.resume_path.resolve()),
                      selected_on="validation", smoke_test=checkpoint["smoke_test"], metrics=metrics)
        write_json(args.resume_path.parent / "test_metrics.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report

    run_dir = args.run_dir or ROOT / "saved1" / (args.model_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Run directory is not empty; choose a new --run-dir.")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update(model_config=config, data_fingerprint=fingerprint,
                      ehr_source="training_patient_cooccurrence", metric_aggregation="patient_macro",
                      training_patients=len(training), validation_patients=len(validation[:args.max_eval_patients]),
                      torch_version=str(torch.__version__), numpy_version=np.__version__, device=str(device))
    write_json(run_dir / "config.json", run_config)
    print("Run directory: {}".format(run_dir.resolve()), flush=True)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                          patience=args.lr_patience, min_lr=1e-6)
                 if args.scheduler == "plateau" else None)
    best, bad_epochs, history = -float("inf"), 0, []
    for epoch in range(args.epochs):
        started = time.time()
        losses, steps = train_epoch(model, training, graphs, optimizer, args, epoch)
        predictions = predict(model, validation[:args.max_eval_patients], graphs)
        fixed, candidates, selected = select_validation(predictions, ddi, args)
        improved = selected is not None and selected["jaccard"] > best
        if improved:
            best, bad_epochs = selected["jaccard"], 0
            payload = dict(format="mnvr_plus_v2",
                           architecture=architecture, model_state=model.state_dict(), model_config=config,
                           epoch=epoch, threshold=selected["threshold"], validation_metrics=selected,
                           validation_fixed_05=fixed, data_fingerprint=fingerprint, seed=args.seed,
                           smoke_test=args.smoke_test, run_config=run_config)
            torch.save(payload, run_dir / "best.pt")
            save_predictions(run_dir / "validation_predictions.npz", predictions)
            write_json(run_dir / "best_validation.json", dict(
                epoch=epoch, architecture=architecture, fixed_05=fixed, selected=selected,
                target=args.target_jaccard,
                target_reached=(best >= args.target_jaccard and not args.smoke_test and args.max_eval_patients is None),
                smoke_test=args.smoke_test, validation_patients=len(predictions)))
        else:
            bad_epochs += 1
        entry = dict(epoch=epoch, training_steps=steps, losses=losses, validation_fixed_05=fixed,
                     validation_selected=selected, threshold_candidates=candidates,
                     improved=improved, lr=optimizer.param_groups[0]["lr"], seconds=time.time() - started)
        if scheduler:
            scheduler.step(selected["jaccard"] if selected else fixed["jaccard"])
        history.append(entry)
        write_json(run_dir / "history.json", history)
        chosen = selected or fixed
        print(("Epoch {epoch} loss={loss:.5f} JA@0.5={fixed:.5f} selected_JA={ja:.5f} "
               "threshold={threshold:.3f} P={precision:.4f} R={recall:.4f} DDI={ddi:.4f} "
               "meds={meds:.2f} lr={lr:.2g} seconds={seconds:.1f} best={best}").format(
                   epoch=epoch, loss=losses["total"], fixed=fixed["jaccard"], ja=chosen["jaccard"],
                   threshold=chosen["threshold"], precision=chosen["precision"], recall=chosen["recall"],
                   ddi=chosen["ddi_rate"], meds=chosen["avg_medications"], lr=entry["lr"],
                   seconds=entry["seconds"], best=improved), flush=True)
        if args.patience > 0 and bad_epochs >= args.patience:
            print("Early stopping after {} epochs without an eligible improvement.".format(bad_epochs), flush=True)
            break
    if best == -float("inf"):
        raise RuntimeError("No checkpoint satisfied the validation DDI constraint; see history.json.")
    print("Best validation Jaccard: {:.6f}; checkpoint: {}".format(best, run_dir / "best.pt"), flush=True)
    print("Test split was not evaluated. Use --eval --resume-path after selecting your experiment.", flush=True)
    return {"run_dir": str(run_dir), "best_validation_jaccard": best}


if __name__ == "__main__":
    main()
