#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for native EasyEDA symbol roles in BOM extraction."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("更新元件库.py")
SPEC = importlib.util.spec_from_file_location("native_library_sync", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load library synchronizer: {SCRIPT_PATH}")
LIBRARY_SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIBRARY_SYNC)


def _section(
    doc_type: str,
    local_uuid: str,
    title: str,
    meta: dict,
    official_uuid: str | None = None,
) -> dict:
    return {
        "doc_type": doc_type,
        "local_uuid": local_uuid,
        "official_uuid": official_uuid,
        "library_uuid": "synthetic-library",
        "title": title,
        "meta": meta,
        "content_sha256": f"synthetic-{doc_type}-{local_uuid}",
    }


class NativeSymbolRolesTest(unittest.TestCase):
    def test_net_flags_and_ports_are_excluded_but_real_no_pcb_part_is_kept(self) -> None:
        sections = [
            _section("SYMBOL", "sym-flag", "GND", {"docType": 18}),
            _section("SYMBOL", "sym-port", "CAN", {"docType": 19}, "official-port"),
            _section("SYMBOL", "sym-component", "Connector", {"docType": 2}),
            _section(
                "DEVICE",
                "dev-flag",
                "GND device",
                {"attributes": {"Add into BOM": "yes", "Convert to PCB": "yes"}},
            ),
            _section(
                "DEVICE",
                "dev-port",
                "CAN port device",
                {"attributes": {"Add into BOM": "yes", "Convert to PCB": "yes"}},
            ),
            _section(
                "DEVICE",
                "dev-component",
                "Abstract interface",
                {"attributes": {"Add into BOM": "yes", "Convert to PCB": "no"}},
            ),
        ]
        records: list[dict] = []

        def add_component(component_id: str, device_id: str, symbol_id: str, attrs: dict) -> None:
            records.append(
                {
                    "header": {"type": "COMPONENT", "id": component_id},
                    "value": {"partId": f"pid-{component_id}", "attrs": {}},
                }
            )
            values = {"Device": device_id, "Symbol": symbol_id, **attrs}
            for index, (key, value) in enumerate(values.items()):
                records.append(
                    {
                        "header": {"type": "ATTR", "id": f"{component_id}-attr-{index}"},
                        "value": {"parentId": component_id, "key": key, "value": value},
                    }
                )

        add_component(
            "flag-instance",
            "dev-flag",
            "sym-flag",
            {
                "Add into BOM": "yes",
                "Convert to PCB": "yes",
                "Designator": "",
                "Manufacturer Part": "",
                "Global Net Name": "GND_SYS",
            },
        )
        # Deliberately reference the official UUID to cover that lookup path.
        add_component(
            "port-instance",
            "dev-port",
            "official-port",
            {
                "Add into BOM": "yes",
                "Convert to PCB": "yes",
                "Designator": "",
                "Manufacturer Part": "",
                "Global Net Name": "CAN_H",
            },
        )
        # Same BOM/global-net attributes, but this is an actual type-2 device.
        add_component(
            "component-instance",
            "dev-component",
            "sym-component",
            {
                "Add into BOM": "yes",
                "Convert to PCB": "no",
                "Designator": "J_ABSTRACT",
                "Manufacturer Part": "",
                "Global Net Name": "INTERFACE_STATUS",
            },
        )

        parsed = {
            "module": "synthetic",
            "path": "synthetic.esch2",
            "sections": sections,
            "pages": [{"records": records, "local_uuid": "page-1", "title": "synthetic"}],
        }

        instances, counts = LIBRARY_SYNC.get_components(parsed, {})

        self.assertEqual(counts["component_count"], 3)
        self.assertEqual(counts["bom_instance_count"], 1)
        self.assertEqual(counts["excluded_non_bom_count"], 2)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["ref"], "J_ABSTRACT")
        self.assertEqual(instances[0]["pcb"], "no")
        self.assertIsNone(instances[0]["mpn"])
        self.assertEqual(instances[0]["attrs"]["effective"]["Global Net Name"], "INTERFACE_STATUS")


if __name__ == "__main__":
    unittest.main()
