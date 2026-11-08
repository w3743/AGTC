"""Windows-native entry wrapper for the frozen Native V2 trainer.

The frozen `train_wsl.py` is started with this wrapper on Windows x64 because
of two host-specific issues:

1. Warp/Newton and PyTorch lazily load CUDA modules. On this sm_120 Windows
   host the mixed lazy loading intermittently raises 0xC0000005 inside
   `NativeFieldEnv.step`. Warp is initialized before the trainer starts and
   cuDNN is disabled (its internal workspaces/streams otherwise overlap with
   Warp host-side CUDA calls).
2. The stock launcher is a POSIX shell script; this wrapper forwards argv and
   keeps the trainer source unchanged except for the documented Windows
   portability patch in `train_wsl.py`.

It does not change physics, observations, rewards or the frozen task identity.
`AGTC_DISABLE_CUDNN=0` and `AGTC_SHARED_STREAM=1` are available as diagnostics;
the stable Windows settings are the defaults.
"""
from __future__ import annotations

import os
import runpy
import sys

FIELD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(FIELD, "native", "train", "train_wsl.py")
TRAIN_DIR = os.path.dirname(TRAIN)
PROTO = os.path.join(FIELD, "native", "proto")
for path in (FIELD, TRAIN_DIR, PROTO):
    if path not in sys.path:
        sys.path.insert(0, path)

if os.environ.get("AGTC_DISABLE_CUDNN", "1") == "1":
    import torch  # noqa: E402

    torch.backends.cudnn.enabled = False
    print("NATIVE_WINDOWS_WRAPPER cudnn disabled", flush=True)

SHARED = os.environ.get("AGTC_SHARED_STREAM", "0") == "1"
torch = None
torch_stream = None
if SHARED:
    import torch  # noqa: E402

    torch.cuda.init()
    torch.cuda.set_device(0)
    torch_stream = torch.cuda.Stream(device=0)
    cuda_stream_handle = int(torch_stream.cuda_stream)

    import warp as wp  # noqa: E402

    wp.init()
    wp.set_device("cuda:0")
    warp_device = wp.get_device("cuda:0")
    warp_stream = wp.Stream(warp_device, cuda_stream=cuda_stream_handle)
    wp.set_stream(warp_stream, sync=True)
    print(f"NATIVE_WINDOWS_WRAPPER shared_stream handle={cuda_stream_handle} "
          f"warp={wp.config.version} device={warp_device}", flush=True)
else:
    import warp as wp  # noqa: E402

    wp.init()
    print(f"NATIVE_WINDOWS_WRAPPER warp={wp.config.version} device={wp.get_device('cuda:0')}",
          flush=True)

if os.environ.get("AGTC_SYNC_CUDA", "0") == "1":
    import torch  # noqa: E402
    from native_env import NativeFieldEnv  # noqa: E402

    _original_step = NativeFieldEnv.step

    def _synchronized_step(self, action):
        torch.cuda.synchronize()
        wp.synchronize()
        result = _original_step(self, action)
        wp.synchronize()
        torch.cuda.synchronize()
        return result

    NativeFieldEnv.step = _synchronized_step
    print("NATIVE_WINDOWS_WRAPPER serialized torch/warp at NativeFieldEnv.step", flush=True)


def _run_trainer() -> None:
    sys.argv = [TRAIN, *sys.argv[1:]]
    runpy.run_path(TRAIN, run_name="__main__")


if SHARED and torch is not None and torch_stream is not None:
    with torch.cuda.stream(torch_stream):
        _run_trainer()
else:
    _run_trainer()
