"""Read installed Newton solver/sensor contact-force path; does not create a simulator."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from newton import Contacts
from newton.sensors import SensorContact


OUT = Path(__file__).resolve().with_name("newton_contact_force_projection_inspection.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excerpts(path: Path, patterns: tuple[str, ...], radius: int = 3) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    hit_indexes = [i for i, line in enumerate(lines) if any(pattern in line for pattern in patterns)]
    windows: list[tuple[int, int]] = []
    for index in hit_indexes:
        start, end = max(0, index - radius), min(len(lines), index + radius + 1)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    return [
        {"start_line": start + 1, "text": "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))}
        for start, end in windows
    ]


def main() -> None:
    contacts = Path(inspect.getfile(Contacts))
    sensor = Path(inspect.getfile(SensorContact))
    newton_root = contacts.parents[2]
    solver_root = newton_root / "_src" / "solvers" / "mujoco"
    solver_files = sorted(solver_root.rglob("*.py"))
    solver_hits = []
    for path in solver_files:
        found = excerpts(path, ("contacts.force", "contact_force", "spatial_vector", "contact_frame"), radius=3)
        if found:
            solver_hits.append({"source": str(path), "sha256": sha256(path), "excerpts": found})
    result = {
        "scope": "read-only installed Newton source inspection; no Isaac app, simulator, GPU, or contact sample created",
        "contacts_source": {
            "source": str(contacts),
            "sha256": sha256(contacts),
            "force_fields": excerpts(contacts, ("currently unused", "self.force: wp.array", "self.force = wp.zeros")),
        },
        "sensor_source": {
            "source": str(sensor),
            "sha256": sha256(sensor),
            "force_decomposition": excerpts(sensor, (
                "force = wp.spatial_top(contact_force[contact_index])",
                "n = wp.normalize(contact_normal[contact_index])",
                "friction = force - wp.dot(force, n) * n",
                "wp.atomic_add(total_force, row0, force)",
                "wp.atomic_add(total_force_friction, row0, friction)",
            ), radius=5),
        },
        "solver_mujoco_source_hits": solver_hits,
    }
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "contacts_sha256": result["contacts_source"]["sha256"],
        "sensor_sha256": result["sensor_source"]["sha256"],
        "solver_files_with_hits": len(solver_hits),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()