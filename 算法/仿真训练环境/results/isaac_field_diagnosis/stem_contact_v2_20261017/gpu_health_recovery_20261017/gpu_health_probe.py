#!/usr/bin/env python3
"""Small CUDA correctness probe; deliberately not a benchmark or simulator run."""

from __future__ import annotations

import json
import platform
import socket
import sys
import traceback
from datetime import datetime, timezone

import numpy as np


result: dict[str, object] = {
    "schema": "isaac-field-gpu-health-probe-v1",
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    },
    "checks": {},
}


def record_failure(stage: str, exc: BaseException) -> None:
    checks = result["checks"]
    assert isinstance(checks, dict)
    checks[stage] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


try:
    import torch

    result["torch"] = {
        "version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; CPU fallback is not accepted")

    torch.cuda.init()
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    result["torch"]["device"] = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
    }

    values = torch.arange(256, dtype=torch.float32, device=device).reshape(16, 16)
    product = values @ values.T
    torch.cuda.synchronize(device)
    copied = product.cpu().numpy()
    if copied.shape != (16, 16) or not np.isfinite(copied).all() or copied[0, 0] != 1240.0:
        raise AssertionError("CUDA tensor operation/copy returned unexpected data")
    checks = result["checks"]
    assert isinstance(checks, dict)
    checks["torch_cuda_tensor_and_d2h"] = {
        "ok": True,
        "device": str(device),
        "shape": list(copied.shape),
        "sample": float(copied[0, 0]),
        "synchronized": True,
    }

    if not torch.backends.cudnn.is_available():
        raise RuntimeError("cuDNN is unavailable; requested convolution check cannot be verified")
    torch.backends.cudnn.enabled = True
    image = torch.randn((2, 3, 16, 16), device=device, requires_grad=True)
    kernel = torch.randn((4, 3, 3, 3), device=device, requires_grad=True)
    conv = torch.nn.functional.conv2d(image, kernel, padding=1)
    loss = conv.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(conv).all().item() or not torch.isfinite(image.grad).all().item() or not torch.isfinite(kernel.grad).all().item():
        raise AssertionError("cuDNN convolution or gradients contain non-finite values")
    checks["cudnn_conv2d_forward_backward"] = {
        "ok": True,
        "input_shape": list(image.shape),
        "weight_shape": list(kernel.shape),
        "output_shape": list(conv.shape),
        "loss": float(loss.detach().cpu()),
        "synchronized": True,
    }
except BaseException as exc:  # record the first torch-stage error and still try Warp independently
    record_failure("torch_cuda_and_cudnn", exc)

try:
    import warp as wp

    result["warp"] = {"version": getattr(wp, "__version__", "unknown")}
    wp.init()
    warp_device = wp.get_device("cuda:0")

    @wp.kernel
    def add_one(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
        i = wp.tid()
        dst[i] = src[i] + 1.0

    source_host = np.arange(64, dtype=np.float32)
    source = wp.array(source_host, dtype=wp.float32, device=warp_device)
    destination = wp.zeros(64, dtype=wp.float32, device=warp_device)
    wp.launch(add_one, dim=64, inputs=[source, destination], device=warp_device)
    wp.synchronize_device(warp_device)
    copied = destination.numpy()
    if copied.shape != (64,) or not np.array_equal(copied, source_host + 1.0):
        raise AssertionError("Warp CUDA array-to-CPU copy returned unexpected data")
    checks = result["checks"]
    assert isinstance(checks, dict)
    checks["warp_cuda_kernel_and_array_to_cpu_copy"] = {
        "ok": True,
        "device": str(warp_device),
        "elements": int(copied.size),
        "first": float(copied[0]),
        "last": float(copied[-1]),
        "synchronized": True,
    }
except BaseException as exc:
    record_failure("warp_cuda_and_d2h", exc)

result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
checks = result["checks"]
assert isinstance(checks, dict)
result["ok"] = bool(checks) and all(isinstance(value, dict) and value.get("ok") is True for value in checks.values())
print(json.dumps(result, indent=2, sort_keys=True))
sys.exit(0 if result["ok"] else 1)
