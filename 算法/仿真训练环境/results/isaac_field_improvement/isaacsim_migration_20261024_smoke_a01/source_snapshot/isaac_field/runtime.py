"""Own one Isaac Sim runtime across training stages and evaluations.

Importing task contracts must remain possible without starting Kit. Native
environments call ``ensure_runtime`` only when they are actually constructed;
multi-stage tools can instead own the whole lifetime with ``launch_app``.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any


SIMULATOR = "isaac_sim"
PHYSICS_BACKEND = "physx"
LEARNING_BACKEND = "rsl_rl"
_runtime = None
_runtime_options: dict[str, Any] | None = None
_atexit_registered = False


def simulator_identity() -> dict[str, str]:
    return {
        "simulator": SIMULATOR,
        "physics_backend": PHYSICS_BACKEND,
        "learning_backend": LEARNING_BACKEND,
    }


def _launcher_options(*, device: str, headless: bool, kit_args: str) -> dict[str, Any]:
    arguments = kit_args.strip()
    if os.name == "nt" and "--portable" not in arguments:
        # MSIX AppData redirection breaks some extension-cache junctions. A
        # short portable root also avoids native DLL MAX_PATH failures on WoA.
        parent = Path(os.environ.get("ISAAC_FIELD_RUNTIME_DIR", "C:/pioneer/isaac-training"))
        # Training, evaluation and preview run serially on this GPU. Reusing
        # this fixed cache avoids recompiling RTX pipelines on every launch.
        portable = parent / 'kit'
        if any(character.isspace() for character in str(portable)):
            raise ValueError("ISAAC_FIELD_RUNTIME_DIR must have no spaces (Kit launcher splits kit_args)")
        portable.mkdir(parents=True, exist_ok=True)
        arguments = f"--portable-root {portable.as_posix()} {arguments}".strip()
    if "/app/settings/fabricDefaultStageFrameHistoryCount" not in arguments:
        arguments += " --/app/settings/fabricDefaultStageFrameHistoryCount=3"
    return {
        "require_kit": True,
        "device": device,
        "headless": bool(headless),
        "enable_cameras": True,
        "kit_args": arguments.strip(),
    }


def ensure_runtime(cfg=None, *, device: str = "cuda:0", headless: bool = True, kit_args: str = ""):
    """Start Kit once, or reuse an app owned by the official Isaac Lab CLI.

    Closing an individual environment must not stop Kit between curriculum
    stages. Tools with an explicit outer lifetime should use ``launch_app``.
    """
    global _runtime, _runtime_options, _atexit_registered
    if os.environ.get('ISAAC_FIELD_DISABLE_NATIVE_RUNTIME') == '1':
        raise RuntimeError('native runtime is disabled for this CPU-only process')
    if _runtime is not None:
        if _runtime_options is not None and str(device) != _runtime_options["device"]:
            raise ValueError("one process cannot silently switch the Isaac Sim runtime device")
        return _runtime
    from isaaclab.utils import has_kit

    if has_kit():
        return None
    from isaaclab.app import launch_simulation

    options = _launcher_options(device=str(device), headless=headless, kit_args=kit_args)
    # This setting is process-local, as in the official non-interactive install.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    runtime = launch_simulation(cfg, options)
    runtime.__enter__()
    _runtime, _runtime_options = runtime, options
    if not _atexit_registered:
        atexit.register(_close_at_exit)
        _atexit_registered = True
    return runtime


def close_runtime(exc_info=None) -> None:
    """Close only the runtime this module owns, preserving exceptional exits."""
    global _runtime, _runtime_options
    runtime, _runtime = _runtime, None
    _runtime_options = None
    if runtime is not None:
        runtime.__exit__(*(exc_info or (None, None, None)))


def _close_at_exit() -> None:
    error = getattr(sys, "last_exc", None)
    close_runtime((type(error), error, error.__traceback__) if error is not None else None)


@contextmanager
def launch_app(cfg=None, **options):
    """Hold a single official runtime around a complete multi-stage workflow."""
    previous = _runtime
    runtime = ensure_runtime(cfg, **options)
    owns_runtime = previous is None and runtime is not None
    try:
        yield runtime
    except BaseException:
        if owns_runtime:
            close_runtime(sys.exc_info())
        raise
    else:
        if owns_runtime:
            close_runtime()


def create_env(*args, **kwargs):
    """Lazily construct the project's native environment with its usual API."""
    from .env import make_env

    return make_env(*args, **kwargs)
