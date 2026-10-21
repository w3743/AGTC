"""CPU-only proof that the local NewtonShapeCfg extension forwards kf to Newton."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from isaaclab.utils import checked_apply, configclass
from isaaclab_newton.physics import NewtonShapeCfg
import newton


@configclass
class NewtonShapeCfgWithKF(NewtonShapeCfg):
    kf: float = 10_000.0


cfg = NewtonShapeCfgWithKF(kf=10_000.0)
builder = newton.ModelBuilder()
checked_apply(cfg, builder.default_shape_cfg)
actual_kf = float(builder.default_shape_cfg.kf)
if actual_kf != 10_000.0:
    raise RuntimeError(f"Newton ModelBuilder.ShapeConfig.kf was {actual_kf}")
probe_path = Path(__file__).with_name("contact_friction_v5.py")
ast.parse(probe_path.read_text(encoding="utf-8"), filename=str(probe_path))
runner_path = Path(__file__).with_name("run_contact_friction_v5.sh")
protocol_path = Path(__file__).with_name("contact_friction_v5_protocol.json")
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
declared_hashes = protocol["source_hashes"]
for key, path in (("v5_helper_sha256", probe_path), ("v5_runner_sha256", runner_path)):
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if actual_hash != declared_hashes[key].upper():
        raise RuntimeError(f"protocol source hash mismatch for {path.name}: {actual_hash}")
if protocol["treatment"]["shape_material_kf_n_s_m"] != 10_000.0:
    raise RuntimeError("protocol does not pin shape_material_kf to 10000 N*s/m")

print(json.dumps({
    "status": "pass",
    "execution_scope": "CPU only; no Simulator, environment, or GPU stepping",
    "wrapper_class": type(cfg).__name__,
    "declared_dataclass_fields": list(cfg.__dataclass_fields__),
    "forwarding_api": "isaaclab.utils.checked_apply",
    "target_type": type(builder.default_shape_cfg).__name__,
    "target_kf_n_s_m": actual_kf,
    "probe_ast": "pass",
    "v5_helper_runner_protocol_hashes": "pass",
}, ensure_ascii=False))
