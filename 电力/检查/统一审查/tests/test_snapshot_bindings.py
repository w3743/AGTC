import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "刷新现行覆盖与成本快照.py"
SPEC = importlib.util.spec_from_file_location("snapshot_refresher", SCRIPT)
snapshot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(snapshot)


class SnapshotBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_power = snapshot.POWER
        self.old_modules = snapshot.MODULES
        self.addCleanup(setattr, snapshot, "POWER", self.old_power)
        self.addCleanup(setattr, snapshot, "MODULES", self.old_modules)

    def configure_module(self, pages):
        project = self.root / "Module" / "design.eprj3"
        project.parent.mkdir(parents=True, exist_ok=True)
        project.write_bytes(b"project")
        configured = []
        for index, page in enumerate(pages):
            record = {"path": f"Module/sch/P{index + 1}.esch2", **page}
            page_path = self.root / record["path"]
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page_path.write_bytes(f"page-{index + 1}".encode())
            configured.append(record)
        snapshot.POWER = self.root
        snapshot.MODULES = [{
            "module": "Test模块",
            "project": "Module/design.eprj3",
            "pages": configured,
        }]
        return configured

    def write_report(self, source_files, **metadata):
        report = self.root / "review.json"
        report.write_text(json.dumps({"sourceFiles": source_files, **metadata}), encoding="utf-8")
        return report

    def source_entry(self, page, path=None, sha=None):
        return {
            "path": path or page["path"],
            "sha256": sha or snapshot.sha256(self.root / page["path"]),
        }

    def test_manifest_uses_moved_bms_path_and_all_four_regulator_pages(self):
        modules = {item["module"]: item for item in snapshot.MODULES}
        bms = modules["BMS模块"]
        self.assertEqual(
            bms["project"],
            "BMS模块/eda工程文件/BMS-9S4P/BMS-9S4P.eprj3",
        )
        self.assertEqual(len(snapshot.module_pages(bms)), 1)
        self.assertEqual(
            snapshot.module_pages(bms)[0]["path"],
            "BMS模块/eda工程文件/BMS-9S4P/sch/Schematic1/P1.esch2",
        )
        self.assertIn("sch/Schematic1/P1.esch2", snapshot.module_pages(bms)[0]["report_path_aliases"])

        regulator_pages = [page["path"] for page in snapshot.module_pages(modules["稳压模块"])]
        self.assertEqual(regulator_pages, [
            "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/01 主功率与超容.esch2",
            "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/02 门极驱动与硬保护.esch2",
            "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/03 模拟采样.esch2",
            "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/04 控制通信与辅助供电.esch2",
        ])
        self.assertIn(
            "sch/sc-regulator/01 主功率与超容.esch2",
            snapshot.module_pages(modules["稳压模块"])[0]["report_path_aliases"],
        )
        self.assertEqual(sum(len(snapshot.module_pages(item)) for item in modules.values()), 7)

    def test_regulator_procurement_uses_current_selected_bom(self):
        sources = [item for item in snapshot.BOM_SOURCES if item["module"] == "稳压模块"]
        self.assertEqual([item["path"] for item in sources], ["稳压模块/原理图准备/选型BOM.csv"])

    def test_complete_multi_page_report_matches_only_each_explicit_page(self):
        self.configure_module([{}, {}, {}, {}])
        pages = snapshot.module_pages(snapshot.MODULES[0])
        report_pages = [
            self.source_entry(page, path=page["report_path_aliases"][-1])
            for page in pages
        ]
        self.write_report(
            report_pages,
            reviewComplete=True,
            designRelease="HOLD",
            status="REVIEW_COMPLETED",
        )

        coverage = snapshot.build_coverage({"Test模块": "review.json"})

        self.assertEqual(coverage["current_page_count"], 4)
        self.assertTrue(coverage["review_evidence_complete"])
        self.assertTrue(coverage["all_review_page_bindings_match"])
        module = coverage["modules"][0]
        self.assertEqual(module["page_count"], 4)
        self.assertEqual(len(module["review"]["page_bindings"]), 4)
        self.assertEqual(module["review"]["page_binding_status"], "MATCH")
        self.assertEqual(module["review"]["review_completion_status"], "COMPLETE")
        self.assertEqual(module["review"]["design_release"], "HOLD")
        self.assertNotIn("page_path", module)

    def test_completion_flags_and_partial_status_are_conservative(self):
        self.assertEqual(snapshot.review_completion_status({"reviewComplete": True, "review_complete": True}), "COMPLETE")
        self.assertEqual(snapshot.review_completion_status({"reviewComplete": True, "status": "PARTIAL_PASS"}), "FAILED")
        self.assertEqual(snapshot.review_completion_status({"reviewComplete": True, "status": "PASS_NOT_COMPLETED"}), "FAILED")
        with self.assertRaises(ValueError):
            snapshot.review_completion_status({"reviewComplete": True, "review_complete": False})
        with self.assertRaises(ValueError):
            snapshot.review_completion_status({"reviewComplete": True, "review_complete": "false"})

    def test_hash_only_report_is_not_evidence_of_review_completion(self):
        pages = self.configure_module([{}, {}, {}, {}])
        self.write_report([self.source_entry(page) for page in pages])

        coverage = snapshot.build_coverage({"Test模块": "review.json"})

        self.assertTrue(coverage["all_review_page_bindings_match"])
        self.assertFalse(coverage["review_evidence_complete"])
        self.assertEqual(coverage["review_completion_unresolved_modules"], ["Test模块"])
        self.assertEqual(
            coverage["modules"][0]["review"]["review_completion_status"],
            "NOT_EXPOSED",
        )

    def test_not_completed_status_blocks_writing_even_with_matching_hashes(self):
        pages = self.configure_module([{}, {}, {}, {}])
        self.write_report(
            [self.source_entry(page) for page in pages],
            reviewComplete=True,
            status="NOT_COMPLETED",
            designRelease="HOLD",
        )

        coverage = snapshot.build_coverage({"Test模块": "review.json"})

        self.assertTrue(coverage["all_review_page_bindings_match"])
        self.assertFalse(coverage["review_evidence_complete"])
        self.assertEqual(
            coverage["modules"][0]["review"]["review_completion_status"],
            "FAILED",
        )
        self.assertEqual(coverage["modules"][0]["review"]["design_release"], "HOLD")

    def test_missing_regulator_page_binding_stays_unresolved(self):
        pages = self.configure_module([{}, {}, {}, {}])
        self.write_report(
            [self.source_entry(page) for page in pages[:3]],
            reviewComplete=True,
            designRelease="HOLD",
        )

        coverage = snapshot.build_coverage({"Test模块": "review.json"})

        self.assertFalse(coverage["review_evidence_complete"])
        self.assertFalse(coverage["all_review_page_bindings_match"])
        self.assertEqual(coverage["missing_or_unresolved_review_modules"], ["Test模块"])
        page_statuses = [entry["status"] for entry in coverage["modules"][0]["review"]["page_bindings"]]
        self.assertEqual(page_statuses, ["MATCH", "MATCH", "MATCH", "NOT_EXPOSED_IN_REPORT"])

    def test_mismatched_page_sha_is_not_accepted(self):
        pages = self.configure_module([{}, {}])
        wrong_sha = "0" * 64
        self.write_report([
            self.source_entry(pages[0]),
            self.source_entry(pages[1], sha=wrong_sha),
        ], reviewComplete=True)

        coverage = snapshot.build_coverage({"Test模块": "review.json"})

        self.assertEqual(coverage["source_binding_mismatches"], ["Test模块"])
        self.assertFalse(coverage["review_evidence_complete"])
        self.assertEqual(
            coverage["modules"][0]["review"]["page_bindings"][1]["status"],
            "MISMATCH",
        )

    def test_bms_old_project_relative_hash_map_requires_unchanged_and_exact_sha(self):
        pages = self.configure_module([{
            "report_path_aliases": ["sch/Schematic1/P1.esch2"],
        }])
        digest = snapshot.sha256(self.root / pages[0]["path"])
        report = {
            "source_bytes_unchanged": True,
            "source_sha256_before": {"sch/Schematic1/P1.esch2": digest},
            "source_sha256_after": {"sch/Schematic1/P1.esch2": digest},
        }
        self.assertEqual(
            snapshot.report_page_binding(report, pages[0]["path"], pages[0]["report_path_aliases"]),
            digest,
        )
        self.assertEqual(
            snapshot.report_page_binding(
                {"sourceFiles": [{"path": "电力/Module/sch/P1.esch2", "sha256": digest}]},
                pages[0]["path"],
                pages[0]["report_path_aliases"],
            ),
            digest,
        )

        report["source_bytes_unchanged"] = False
        self.assertIsNone(
            snapshot.report_page_binding(report, pages[0]["path"], pages[0]["report_path_aliases"]),
        )

    def test_conflicting_current_page_hashes_abort_binding(self):
        pages = self.configure_module([{}])
        digest = snapshot.sha256(self.root / pages[0]["path"])
        report = {
            "sourceFiles": [self.source_entry(pages[0])],
            "sourceBinding": {
                "nativePath": pages[0]["path"],
                "nativeSha256": "f" * 64 if digest != "f" * 64 else "e" * 64,
            },
        }

        with self.assertRaises(ValueError):
            snapshot.report_page_binding(report, pages[0]["path"])

    def test_legacy_single_page_report_does_not_bind_a_new_multi_page_set(self):
        pages = self.configure_module([{}, {}, {}, {}])
        legacy = {
            "path": "Module/sch/01 旧的合并页.esch2",
            "sha256": "a" * 64,
        }
        self.write_report([legacy], reviewComplete=True)

        coverage = snapshot.build_coverage({"Test模块": "review.json"})

        self.assertFalse(coverage["all_review_page_bindings_match"])
        statuses = [entry["status"] for entry in coverage["modules"][0]["review"]["page_bindings"]]
        self.assertEqual(statuses, ["NOT_EXPOSED_IN_REPORT"] * 4)


if __name__ == "__main__":
    unittest.main()
