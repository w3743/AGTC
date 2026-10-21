"""Read-only inspection of installed Newton contact APIs; does not create a simulator."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import newton
from newton import Contacts
from newton.sensors import SensorContact


OUT = Path(__file__).resolve().with_name("newton_contact_api_inspection.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_hits(source: Path, patterns: tuple[str, ...]) -> list[dict[str, object]]:
    found = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if any(pattern in line for pattern in patterns):
            found.append({"line": number, "text": line.strip()})
    return found


def main() -> None:
    contacts_source = Path(inspect.getfile(Contacts))
    sensor_source = Path(inspect.getfile(SensorContact))
    result = {
        "scope": "source/API inspection only; no Isaac app, simulator, GPU, or contact sample created",
        "newton_version": getattr(newton, "__version__", None),
        "contacts": {
            "source": str(contacts_source),
            "sha256": sha256(contacts_source),
            "identity_fields": line_hits(contacts_source, (
                "self.rigid_contact_count =",
                "self.rigid_contact_shape0 =",
                "self.rigid_contact_shape1 =",
                "self.rigid_contact_normal =",
            )),
            "force_fields": line_hits(contacts_source, (
                "self.rigid_contact_force =",
                "currently unused",
                "self.force: wp.array | None = None",
                "self.force = wp.zeros(total_contacts",
                '"force" in requested_attributes',
            )),
        },
        "sensor_contact": {
            "source": str(sensor_source),
            "sha256": sha256(sensor_source),
            "force_aggregation": line_hits(sensor_source, (
                "force = wp.spatial_top(contact_force[contact_index])",
                "friction = force - wp.dot(force, n) * n",
                "wp.atomic_add(total_force, row0, force)",
                "wp.atomic_add(total_force_friction, row0, friction)",
                "Total friction (tangential) contact force [N]",
                "total_force — the total contact force on each sensing object",
            )),
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "newton_version": result["newton_version"],
                      "contacts_sha256": result["contacts"]["sha256"],
                      "sensor_contact_sha256": result["sensor_contact"]["sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
