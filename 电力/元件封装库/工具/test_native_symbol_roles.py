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
    def test_net_flags_ports_and_offpage_are_excluded_but_real_no_pcb_part_is_kept(self) -> None:
        sections = [
            _section("SYMBOL", "sym-flag", "GND", {"docType": 18}),
            _section("SYMBOL", "sym-port", "CAN", {"docType": 19}, "official-port"),
            _section("SYMBOL", "sym-offpage", "OFFPAGE_OUT", {"docType": 25}),
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
                "dev-offpage",
                "Off-page connector",
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
        add_component(
            "offpage-instance",
            "dev-offpage",
            "sym-offpage",
            {
                "Add into BOM": "yes",
                "Convert to PCB": "yes",
                "Designator": "",
                "Manufacturer Part": "",
                "Global Net Name": "BUS_EXT",
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

        self.assertEqual(counts["component_count"], 4)
        self.assertEqual(counts["bom_instance_count"], 1)
        self.assertEqual(counts["excluded_non_bom_count"], 3)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["ref"], "J_ABSTRACT")
        self.assertEqual(instances[0]["pcb"], "no")
        self.assertIsNone(instances[0]["mpn"])
        self.assertEqual(instances[0]["attrs"]["effective"]["Global Net Name"], "INTERFACE_STATUS")


class NativeMultiPageStatsTest(unittest.TestCase):
    def test_same_module_sources_aggregate_stats_but_keep_per_source_evidence(self) -> None:
        sources = [
            {"module": "稳压模块", "path": "regulator/page1.esch2"},
            {"module": "稳压模块", "path": "regulator/page2.esch2"},
        ]
        source_stats = {
            ("稳压模块", "regulator/page1.esch2"): {
                "component_count": 54,
                "bom_instance_count": 54,
                "excluded_non_bom_count": 5,
                "missing_device_binding_count": 1,
                "missing_symbol_binding_count": 0,
                "missing_footprint_binding_count": 0,
            },
            ("稳压模块", "regulator/page2.esch2"): {
                "component_count": 46,
                "bom_instance_count": 46,
                "excluded_non_bom_count": 3,
                "missing_device_binding_count": 0,
                "missing_symbol_binding_count": 1,
                "missing_footprint_binding_count": 2,
            },
        }

        module_order = LIBRARY_SYNC.unique_module_order(sources)
        module_stats = LIBRARY_SYNC.aggregate_module_stats(source_stats, module_order)
        parsed_sources = [
            {
                **source,
                "sha256_before": before,
                "sha256_after_parse": before,
                "pages": [{"title": "page"}],
            }
            for source, before in zip(sources, ("a" * 64, "b" * 64))
        ]
        evidence_rows = LIBRARY_SYNC.build_source_evidence_rows(
            parsed_sources, ["a" * 64, "b" * 64], source_stats
        )

        self.assertEqual(module_order, ["稳压模块"])
        self.assertEqual(len(module_stats), 1)
        self.assertEqual(module_stats[0]["component_count"], 100)
        self.assertEqual(module_stats[0]["bom_instance_count"], 100)
        self.assertEqual(module_stats[0]["excluded_non_bom_count"], 8)
        self.assertEqual([row["component_count"] for row in evidence_rows], [54, 46])
        self.assertEqual([row["bom_instance_count"] for row in evidence_rows], [54, 46])
        self.assertEqual([row["path"] for row in evidence_rows], [source["path"] for source in sources])
        self.assertEqual([row["sha256"] for row in evidence_rows], ["a" * 64, "b" * 64])
        self.assertTrue(all(row["unchanged"] for row in evidence_rows))


if __name__ == "__main__":
    unittest.main()
