"""Reproducible MNIST trainer for the local-neuron model and an MLP reference.

The official test set is opt-in and is loaded only after training finishes.
Normal runs use a fixed, disjoint train/dev subset of the official training
split. Results are written to a new, never-reused run directory outside Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import data as mnist_data


DEFAULT_RUNS_ROOT = Path("C:/pioneer/local-neuron-runs")
SOURCE_FILES = ("data.py", "baseline.py", "train.py", "model.py", "requirements.txt")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n").encode("utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def create_run_dir(runs_root: Path) -> tuple[Path, str]:
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(50):
        run_id = f"mnist_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}_{uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir, run_id
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique run directory in {runs_root}")


def capture_sources(run_dir: Path) -> dict[str, dict[str, str]]:
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    for name in SOURCE_FILES:
        source = source_dir / name
        if not source.is_file():
            manifest[name] = {"status": "missing"}
            continue
        payload = source.read_bytes()
        destination = snapshot_dir / name
        destination.write_bytes(payload)
        manifest[name] = {
            "status": "captured",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": str(len(payload)),
        }
    write_json_atomic(run_dir / "source_hashes.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("local", "baseline"), default="local")
    parser.add_argument("--seed", type=int, default=1, help="Model initialization and per-epoch order seed")
    parser.add_argument("--split-seed", type=int, default=20261026)
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--dev-count", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0")
    parser.add_argument("--data-cache", type=Path, default=mnist_data.DEFAULT_CACHE_DIR)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--eval-official-test", action="store_true", help="Explicitly evaluate all 10,000 official test examples after training")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Use generated tiny data to check the runner without downloading MNIST")

    parser.add_argument("--baseline-hidden", type=int, default=256)
    parser.add_argument("--baseline-second-hidden", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--input-degree", type=int, default=32)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--memory-dim", type=int, default=8)
    parser.add_argument("--edge-hidden", type=int, default=32)
    parser.add_argument("--cell-hidden", type=int, default=32)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.train_count <= 0 or args.dev_count <= 0:
        parser.error("--train-count and --dev-count must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate must be positive and --weight-decay cannot be negative")
    if args.eval_official_test and args.synthetic_smoke:
        parser.error("--eval-official-test cannot be combined with --synthetic-smoke")
    return args


def args_payload(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def configure_torch(seed: int) -> tuple[Any, Any]:
    # Set before importing torch so cuBLAS can use its deterministic workspace.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    import warnings

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return torch, warnings


def environment_payload(torch: Any, device: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "deterministic_algorithms": "enabled_warn_only",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device.type == "cuda":
        payload["device_name"] = torch.cuda.get_device_name(device)
        payload["device_capability"] = list(torch.cuda.get_device_capability(device))
        payload["cuda_device_count"] = torch.cuda.device_count()
    return payload


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested}, but CUDA is unavailable in this PyTorch environment")
    return device


def make_model(torch: Any, args: argparse.Namespace) -> Any:
    if args.model == "baseline":
        from baseline import MLPBaseline

        return MLPBaseline(args.baseline_hidden, args.baseline_second_hidden)

    try:
        from model import LocalNeuronClassifier
    except ImportError as exc:
        raise RuntimeError("--model local requires model.py exporting LocalNeuronClassifier") from exc
    return LocalNeuronClassifier(
        seed=args.seed,
        n_hidden=args.n_hidden,
        input_degree=args.input_degree,
        recurrent_degree=args.recurrent_degree,
        steps=args.steps,
        memory_dim=args.memory_dim,
        edge_hidden=args.edge_hidden,
        cell_hidden=args.cell_hidden,
    )


def make_optimizer(torch: Any, args: argparse.Namespace, model: Any) -> Any:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def tensorize(torch: Any, images_raw: bytes, labels_raw: bytes, count: int) -> tuple[Any, Any]:
    images = torch.frombuffer(bytearray(images_raw), dtype=torch.uint8).reshape(count, 28 * 28)
    labels = torch.frombuffer(bytearray(labels_raw), dtype=torch.uint8).to(dtype=torch.long)
    return images, labels


def prepare_dataset(torch: Any, args: argparse.Namespace, run_dir: Path) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    if args.synthetic_smoke:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 8101)
        train_x = torch.randint(0, 256, (min(args.train_count, 256), 784), dtype=torch.uint8, generator=generator)
        train_y = torch.randint(0, 10, (train_x.shape[0],), dtype=torch.long, generator=generator)
        dev_x = torch.randint(0, 256, (min(args.dev_count, 64), 784), dtype=torch.uint8, generator=generator)
        dev_y = torch.randint(0, 10, (dev_x.shape[0],), dtype=torch.long, generator=generator)
        synthetic = {
            "kind": "synthetic_smoke_only",
            "generator_seed": args.seed + 8101,
            "train_sha256": hashlib.sha256(train_x.numpy().tobytes() + train_y.numpy().tobytes()).hexdigest(),
            "dev_sha256": hashlib.sha256(dev_x.numpy().tobytes() + dev_y.numpy().tobytes()).hexdigest(),
        }
        write_json_atomic(run_dir / "dataset.json", synthetic)
        return train_x, train_y, dev_x, dev_y, synthetic

    train_data = mnist_data.load_mnist(cache_dir=args.data_cache, split="train")
    all_images, all_labels = tensorize(torch, train_data.images, train_data.labels, train_data.count)
    train_indices, dev_indices = mnist_data.make_fixed_split_indices(
        total_count=train_data.count,
        train_count=args.train_count,
        dev_count=args.dev_count,
        seed=args.split_seed,
    )
    split_info = mnist_data.split_payload(train_indices, dev_indices, args.split_seed)
    split_sha256 = mnist_data.write_split_indices(run_dir / "split_indices.json", split_info)
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    dev_index_tensor = torch.tensor(dev_indices, dtype=torch.long)
    train_x = all_images.index_select(0, train_index_tensor)
    train_y = all_labels.index_select(0, train_index_tensor)
    dev_x = all_images.index_select(0, dev_index_tensor)
    dev_y = all_labels.index_select(0, dev_index_tensor)
    dataset_info = {
        "kind": "official_mnist_train_split",
        "official_test_loaded": False,
        "official_train": train_data.hashes(),
        "split_file": str((run_dir / "split_indices.json").resolve()),
        "split_sha256": split_sha256,
        "train_count": len(train_indices),
        "dev_count": len(dev_indices),
    }
    write_json_atomic(run_dir / "dataset.json", dataset_info)
    del all_images, all_labels
    return train_x, train_y, dev_x, dev_y, dataset_info


def sync_device(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batch_images(torch: Any, images: Any, indices: Any, device: Any) -> Any:
    batch = images.index_select(0, indices).to(device=device, non_blocking=False)
    return batch.to(dtype=torch.float32).div_(255.0).reshape(-1, 1, 28, 28)


def train_epoch(torch: Any, model: Any, optimizer: Any, images: Any, labels: Any, args: argparse.Namespace, device: Any, order_seed: int) -> dict[str, float | int]:
    model.train()
    generator = torch.Generator(device="cpu").manual_seed(order_seed)
    order = torch.randperm(images.shape[0], generator=generator)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct_sum = torch.zeros((), device=device, dtype=torch.long)
    steps = 0
    start = time.perf_counter()
    sync_device(torch, device)
    for start_index in range(0, images.shape[0], args.batch_size):
        batch_indices = order[start_index : start_index + args.batch_size]
        batch_x = batch_images(torch, images, batch_indices, device)
        batch_y = labels.index_select(0, batch_indices).to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x, plastic=True)
        loss = torch.nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()
        count = batch_y.shape[0]
        loss_sum += loss.detach().to(dtype=torch.float64) * count
        correct_sum += (logits.detach().argmax(dim=1) == batch_y).sum()
        steps += 1
    sync_device(torch, device)
    elapsed = time.perf_counter() - start
    return {
        "loss": float((loss_sum / images.shape[0]).item()),
        "accuracy": float((correct_sum / images.shape[0]).item()),
        "steps": steps,
        "elapsed_seconds": elapsed,
    }


def evaluate(torch: Any, model: Any, images: Any, labels: Any, batch_size: int, device: Any) -> dict[str, Any]:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct_sum = torch.zeros((), device=device, dtype=torch.long)
    predictions: list[int] = []
    confidences: list[float] = []
    start = time.perf_counter()
    sync_device(torch, device)
    with torch.no_grad():
        for start_index in range(0, images.shape[0], batch_size):
            indices = torch.arange(start_index, min(start_index + batch_size, images.shape[0]), dtype=torch.long)
            batch_x = batch_images(torch, images, indices, device)
            batch_y = labels.index_select(0, indices).to(device=device, dtype=torch.long)
            logits = model(batch_x, plastic=True)
            loss_sum += torch.nn.functional.cross_entropy(logits, batch_y, reduction="sum").to(dtype=torch.float64)
            predicted = logits.argmax(dim=1)
            correct_sum += (predicted == batch_y).sum()
            probs = torch.softmax(logits, dim=1)
            confidences.extend(probs.max(dim=1).values.detach().cpu().tolist())
            predictions.extend(predicted.detach().cpu().tolist())
    sync_device(torch, device)
    elapsed = time.perf_counter() - start
    return {
        "loss": float((loss_sum / images.shape[0]).item()),
        "accuracy": float((correct_sum / images.shape[0]).item()),
        "count": int(images.shape[0]),
        "elapsed_seconds": elapsed,
        "predictions": predictions,
        "confidences": confidences,
    }


def confusion_matrix(truth: list[int], predictions: list[int]) -> list[list[int]]:
    if len(truth) != len(predictions):
        raise ValueError("truth and prediction lengths differ")
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    for actual, predicted in zip(truth, predictions):
        if not 0 <= actual <= 9 or not 0 <= predicted <= 9:
            raise ValueError("label/prediction outside 0..9")
        matrix[actual][predicted] += 1
    return matrix


def save_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\prediction", *range(10)])
        for digit, row in enumerate(matrix):
            writer.writerow([digit, *row])


def save_predictions_csv(path: Path, labels: Any, result: dict[str, Any]) -> None:
    truths = labels.tolist()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["official_test_index", "truth", "prediction", "confidence", "correct"])
        for index, (truth, prediction, confidence) in enumerate(
            zip(truths, result["predictions"], result["confidences"])
        ):
            writer.writerow([index, truth, prediction, f"{confidence:.9g}", int(truth == prediction)])


def model_structure_summary(model: Any) -> dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    summary: dict[str, Any] = {"trainable_parameters": int(trainable), "total_parameters": int(total)}
    state = model.state_dict()
    edge_tensors: dict[str, Any] = {}
    for name, value in state.items():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in {"edge_src", "edge_dst", "edge_kind"} and hasattr(value, "numel"):
            edge_tensors[suffix] = value.detach().cpu()
    if "edge_src" in edge_tensors:
        summary["edge_count"] = int(edge_tensors["edge_src"].numel())
    if "edge_dst" in edge_tensors:
        summary["edge_destination_count"] = int(edge_tensors["edge_dst"].numel())
    if "edge_kind" in edge_tensors:
        kinds = edge_tensors["edge_kind"].tolist()
        counts: dict[str, int] = {}
        for kind in kinds:
            key = str(kind)
            counts[key] = counts.get(key, 0) + 1
        summary["edge_kind_counts"] = counts
    for method_name in ("topology_summary", "structure_summary"):
        method = getattr(model, method_name, None)
        if callable(method):
            try:
                summary["model_summary"] = method()
                break
            except Exception as exc:  # Keep a summary-hook error visible without losing the run.
                summary["summary_hook_error"] = f"{method_name}: {type(exc).__name__}: {exc}"
    return summary


def drain_structure_events(model: Any) -> list[Any]:
    for method_name in ("drain_structure_events", "pop_structure_events"):
        method = getattr(model, method_name, None)
        if callable(method):
            events = method()
            return events if isinstance(events, list) else [events]
    return []


def save_checkpoint(torch: Any, path: Path, model: Any, optimizer: Any, args: argparse.Namespace, epoch: int, metric: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "seed": args.seed,
        "model": args.model,
        "config": args_payload(args),
        "dev_metric": {key: value for key, value in metric.items() if key not in ("predictions", "confidences")},
    }
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def copy_best_state(torch: Any, source_path: Path, model: Any, device: Any) -> dict[str, Any]:
    checkpoint = torch.load(source_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_experiment(args: argparse.Namespace, run_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    torch, _warnings = configure_torch(args.seed)
    device = resolve_device(torch, args.device)
    write_json_atomic(run_dir / "environment.json", environment_payload(torch, device))
    status.update({"phase": "prepare_data", "updated_at": utc_now(), "device": str(device)})
    write_json_atomic(run_dir / "run_status.json", status)

    train_x, train_y, dev_x, dev_y, dataset_info = prepare_dataset(torch, args, run_dir)
    if args.eval_official_test:
        # The official test split is untouched until this explicit post-training phase.
        test_data = None
    else:
        test_data = None

    order_seed_base = args.seed
    model = make_model(torch, args).to(device)
    optimizer = make_optimizer(torch, args, model)
    initial_structure = model_structure_summary(model)
    append_jsonl(
        run_dir / "structure_events.jsonl",
        {"event": "initial_topology_snapshot", "epoch": 0, "summary": initial_structure, "model_events": drain_structure_events(model)},
    )
    trainable_parameters = initial_structure["trainable_parameters"]
    total_parameters = initial_structure["total_parameters"]
    total_steps = 0
    epoch_records: list[dict[str, Any]] = []
    best_epoch = 0
    best_metric: tuple[float, float] | None = None
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    training_start = time.perf_counter()
    status.update({"phase": "training", "updated_at": utc_now(), "epochs_completed": 0})
    write_json_atomic(run_dir / "run_status.json", status)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        order_seed = order_seed_base + 1_000_003 * epoch
        train_result = train_epoch(torch, model, optimizer, train_x, train_y, args, device, order_seed)
        dev_result = evaluate(torch, model, dev_x, dev_y, args.batch_size, device)
        sync_device(torch, device)
        epoch_total = time.perf_counter() - epoch_start
        total_steps += int(train_result["steps"])
        record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "dev_loss": dev_result["loss"],
            "dev_accuracy": dev_result["accuracy"],
            "train_steps": train_result["steps"],
            "cumulative_steps": total_steps,
            "order_seed": order_seed,
            "train_seconds": train_result["elapsed_seconds"],
            "dev_seconds": dev_result["elapsed_seconds"],
            "epoch_seconds": epoch_total,
        }
        append_jsonl(run_dir / "epoch_metrics.jsonl", record)
        epoch_records.append(record)

        candidate_metric = (float(dev_result["accuracy"]), -float(dev_result["loss"]))
        if best_metric is None or candidate_metric > best_metric:
            best_metric = candidate_metric
            best_epoch = epoch
            save_checkpoint(torch, best_checkpoint_path, model, optimizer, args, epoch, dev_result)
        save_checkpoint(torch, run_dir / "last_checkpoint.pt", model, optimizer, args, epoch, dev_result)
        topology = model_structure_summary(model)
        model_events = drain_structure_events(model)
        append_jsonl(
            run_dir / "structure_events.jsonl",
            {"event": "epoch_topology_snapshot", "epoch": epoch, "summary": topology, "model_events": model_events},
        )
        status.update(
            {
                "phase": "training",
                "updated_at": utc_now(),
                "epochs_completed": epoch,
                "latest_train_loss": record["train_loss"],
                "latest_train_accuracy": record["train_accuracy"],
                "latest_dev_loss": record["dev_loss"],
                "latest_dev_accuracy": record["dev_accuracy"],
            }
        )
        write_json_atomic(run_dir / "run_status.json", status)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    sync_device(torch, device)
    training_seconds = time.perf_counter() - training_start
    best_checkpoint = copy_best_state(torch, best_checkpoint_path, model, device)
    append_jsonl(
        run_dir / "structure_events.jsonl",
        {"event": "best_checkpoint_selected", "epoch": best_epoch, "summary": model_structure_summary(model)},
    )
    final_dev = evaluate(torch, model, dev_x, dev_y, args.batch_size, device)
    save_confusion_csv(run_dir / "dev_confusion_matrix.csv", confusion_matrix(dev_y.tolist(), final_dev["predictions"]))

    report: dict[str, Any] = {
        "status": "completed",
        "run_id": run_dir.name,
        "model": args.model,
        "seed": args.seed,
        "config": args_payload(args),
        "dataset": dataset_info,
        "parameter_count": {"trainable": trainable_parameters, "total": total_parameters},
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(epoch_records),
            "best_epoch_by_dev": best_epoch,
            "best_dev_accuracy": float(best_checkpoint["dev_metric"]["accuracy"]),
            "best_dev_loss": float(best_checkpoint["dev_metric"]["loss"]),
            "selected_dev_accuracy": final_dev["accuracy"],
            "selected_dev_loss": final_dev["loss"],
            "train_examples_per_epoch": int(train_x.shape[0]),
            "dev_examples": int(dev_x.shape[0]),
            "batch_size": args.batch_size,
            "steps_total": total_steps,
            "steps_per_epoch": [int(row["train_steps"]) for row in epoch_records],
            "elapsed_seconds": training_seconds,
            "epoch_seconds": [float(row["epoch_seconds"]) for row in epoch_records],
        },
        "development": {
            "loss": final_dev["loss"],
            "accuracy": final_dev["accuracy"],
            "count": final_dev["count"],
        },
        "official_test": {"evaluated": False},
        "baseline_comparison_note": "Parameter counts, optimization steps and wall time are reported separately; this experiment does not claim compute-matched fairness.",
        "artifacts": {
            "best_checkpoint": str(best_checkpoint_path.resolve()),
            "last_checkpoint": str((run_dir / "last_checkpoint.pt").resolve()),
            "epoch_metrics": str((run_dir / "epoch_metrics.jsonl").resolve()),
            "structure_events": str((run_dir / "structure_events.jsonl").resolve()),
            "dev_confusion_matrix": str((run_dir / "dev_confusion_matrix.csv").resolve()),
        },
    }

    if args.eval_official_test:
        status.update({"phase": "official_test_evaluation", "updated_at": utc_now()})
        write_json_atomic(run_dir / "run_status.json", status)
        test_data = mnist_data.load_mnist(cache_dir=args.data_cache, split="test")
        test_x, test_y = tensorize(torch, test_data.images, test_data.labels, test_data.count)
        test_result = evaluate(torch, model, test_x, test_y, args.batch_size, device)
        test_matrix = confusion_matrix(test_y.tolist(), test_result["predictions"])
        save_confusion_csv(run_dir / "official_test_confusion_matrix.csv", test_matrix)
        save_predictions_csv(run_dir / "official_test_predictions.csv", test_y, test_result)
        report["dataset"]["official_test"] = test_data.hashes()
        report["official_test"] = {
            "evaluated": True,
            "count": test_result["count"],
            "loss": test_result["loss"],
            "accuracy": test_result["accuracy"],
            "elapsed_seconds": test_result["elapsed_seconds"],
            "confusion_matrix": test_matrix,
            "predictions_csv": str((run_dir / "official_test_predictions.csv").resolve()),
            "confusion_matrix_csv": str((run_dir / "official_test_confusion_matrix.csv").resolve()),
        }
        report["artifacts"]["official_test_predictions"] = str((run_dir / "official_test_predictions.csv").resolve())
        report["artifacts"]["official_test_confusion_matrix"] = str((run_dir / "official_test_confusion_matrix.csv").resolve())

    write_json_atomic(run_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir, run_id = create_run_dir(args.runs_root)
    started_at = utc_now()
    status: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "phase": "startup",
        "started_at": started_at,
        "updated_at": started_at,
        "exit_code": None,
        "config": args_payload(args),
    }
    write_json_atomic(run_dir / "run_status.json", status)
    start = time.perf_counter()
    try:
        source_manifest = capture_sources(run_dir)
        config = args_payload(args)
        config["run_id"] = run_id
        config["source_hashes"] = source_manifest
        config_path = run_dir / "config.json"
        write_json_atomic(config_path, config)
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        status["config_sha256"] = config_sha
        status["phase"] = "initializing"
        status["updated_at"] = utc_now()
        write_json_atomic(run_dir / "run_status.json", status)

        report = run_experiment(args, run_dir, status)
        total_elapsed = time.perf_counter() - start
        status.update(
            {
                "status": "completed",
                "phase": "finished",
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "exit_code": 0,
                "elapsed_seconds": total_elapsed,
                "report": str((run_dir / "report.json").resolve()),
            }
        )
        report["elapsed_wall_seconds_including_setup"] = total_elapsed
        write_json_atomic(run_dir / "report.json", report)
        write_json_atomic(run_dir / "run_status.json", status)
        print(json.dumps({"run_dir": str(run_dir.resolve()), "report": report}, ensure_ascii=False, sort_keys=True, default=str))
        return 0
    except KeyboardInterrupt:
        error_text = "KeyboardInterrupt"
        exit_code = 130
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        exit_code = 1

    (run_dir / "error.txt").write_text(error_text, encoding="utf-8")
    total_elapsed = time.perf_counter() - start
    status.update(
        {
            "status": "failed",
            "phase": "failed",
            "updated_at": utc_now(),
            "finished_at": utc_now(),
            "exit_code": exit_code,
            "elapsed_seconds": total_elapsed,
            "error": error_text,
        }
    )
    write_json_atomic(run_dir / "run_status.json", status)
    write_json_atomic(
        run_dir / "report.json",
        {
            "status": "failed",
            "run_id": run_id,
            "exit_code": exit_code,
            "config": args_payload(args),
            "elapsed_wall_seconds_including_setup": total_elapsed,
            "error": error_text,
            "artifacts": {"run_status": str((run_dir / "run_status.json").resolve()), "error": str((run_dir / "error.txt").resolve())},
        },
    )
    print(f"MNIST run failed; evidence saved in {run_dir}: {error_text.splitlines()[0]}", file=sys.stderr, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
