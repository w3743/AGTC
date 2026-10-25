"""Standard-library V2 protocol/recipe gates; no simulator or checkpoint deserialization."""
from __future__ import annotations
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TRAIN = Path(__file__).resolve().parents[1] / 'train'
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))
import trial_checks
import protocol_sha
from contract_identity import attach_contract_identity
from test_task_contract import newton_mujoco_warp_physics
from test_trial_runners import contract_fixture, write_clean_evidence


def write_v2_evidence(trial: Path, digest: str):
    write_clean_evidence(trial, digest)
    path = trial / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["physics_solver_events"] = {"world_substeps": 102400, "capacity_overflow_substeps": 0,
                                         "outer_iteration_limit_substeps": 0, "line_search_limit_substeps": 0}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class V2ProtocolGateTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.temp = tempfile.TemporaryDirectory(prefix='agtc v2 evidence ')
        self.addCleanup(self.temp.cleanup)
        self.trial = Path(self.temp.name) / 'trial'
        self.trial.mkdir()
        self.protocol = self.trial / 'PROTOCOL_NATIVE_V2.md'
        self.protocol.write_text('# V2 fixture\n> protocol_sha256:\nFixed fixture body.\n', encoding='utf-8')
        digest = protocol_sha.compute(self.protocol)
        protocol_sha.embed(digest, self.protocol)
        self.sha_file = self.protocol.with_suffix('.sha256')
        self.sha_file.write_text(json.dumps({'file': self.protocol.name, 'protocol_sha256': digest}), encoding='utf-8')
        self.contract, self.identity = contract_fixture()
        physics = newton_mujoco_warp_physics()
        self.contract['physics'] = physics
        attach_contract_identity(self.contract)
        self.identity['physics'] = 'newton_mujoco_warp'
        self.identity['task_identity'] = self.contract['task_identity']
        self.identity['task_identity_sha256'] = self.contract['task_identity_sha256']
        self.identity['env']['physics'] = copy.deepcopy(physics)
        self.identity['env']['task_identity_sha256'] = self.contract['task_identity_sha256']
        self.save_sidecars()
        self.checkpoint = self.trial / 'latest.pt'
        self.checkpoint.write_bytes(b'opaque fixture checkpoint: payload validation is a separate actual-stack check')
        write_v2_evidence(self.trial, self.contract['task_identity_sha256'])
        summary_path = self.trial / 'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['physics'] = 'newton_mujoco_warp'
        summary_path.write_text(json.dumps(summary), encoding='utf-8')
        self.profile = {'sha': digest, 'task_sha': self.contract['task_identity_sha256'],
                        'engine': 'newton_mujoco_warp', 'phase': 'formal_native_v2_trial'}
        # Test the gate with a miniature frozen protocol rather than pretending this
        # synthetic contract is the real machine's validated V2 task fingerprint.
        self.profile_patch = patch.dict(trial_checks.PROTOCOL_PROFILES, {'V2': self.profile})
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)

    def save_sidecars(self):
        for filename, payload in (('observation_contract.json', self.contract), ('identity.json', self.identity)):
            (self.trial / filename).write_text(json.dumps(payload), encoding='utf-8')

    def test_v2_complete_recipe_writes_manifest_and_preflights(self):
        self.assertEqual(trial_checks.write_manifest(self.trial, 0, self.sha_file, 'V2'), 0)
        manifest = json.loads((self.trial / 'trial_manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(manifest['formal_eligible'])
        self.assertEqual(manifest['protocol_version'], 'V2')
        self.assertEqual(manifest['phase'], 'formal_native_v2_trial')
        out = Path(self.temp.name) / 'evaluation'
        self.assertEqual(trial_checks.preflight_checkpoint(self.checkpoint, out, self.sha_file, 'V2'), 0)
        proof = json.loads((out / 'checkpoint_preflight.json').read_text(encoding='utf-8'))
        self.assertEqual(proof['protocol_version'], 'V2')
        self.assertEqual(proof['task_identity_sha256'], self.profile['task_sha'])

    def test_v2_rejects_another_valid_task_fingerprint(self):
        self.contract['task']['reward_termination']['time_penalty'] = 0.02
        attach_contract_identity(self.contract)
        self.identity['task_identity'] = self.contract['task_identity']
        self.identity['task_identity_sha256'] = self.contract['task_identity_sha256']
        self.identity['env']['task_identity_sha256'] = self.contract['task_identity_sha256']
        self.save_sidecars()
        write_v2_evidence(self.trial, self.contract['task_identity_sha256'])
        summary_path = self.trial / 'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['physics'] = 'newton_mujoco_warp'
        summary_path.write_text(json.dumps(summary), encoding='utf-8')
        self.assertEqual(trial_checks.write_manifest(self.trial, 0, self.sha_file, 'V2'), 3)
        manifest = json.loads((self.trial / 'trial_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['validation_errors'], ['task identity differs from the frozen protocol recipe'])

    def test_exit_zero_with_incomplete_or_truncated_physics_evidence_is_rejected(self):
        cases = (
            ('missing', 'is missing physics solver diagnostics'),
            ('capacity', 'reports physics capacity truncation'),
            ('coverage', 'has incomplete physics solver diagnostics'),
        )
        for case, message in cases:
            with self.subTest(case=case):
                write_v2_evidence(self.trial, self.contract['task_identity_sha256'])
                summary_path = self.trial / 'summary.json'
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                summary['physics'] = 'newton_mujoco_warp'
                summary_path.write_text(json.dumps(summary), encoding='utf-8')
                metrics_path = self.trial / 'metrics.jsonl'
                rows = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines()]
                if case == 'missing':
                    del rows[3]['physics_solver_events']
                elif case == 'capacity':
                    rows[3]['physics_solver_events']['capacity_overflow_substeps'] = 1
                else:
                    rows[3]['physics_solver_events']['world_substeps'] -= 50
                metrics_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                self.assertEqual(trial_checks.write_manifest(self.trial, 0, self.sha_file, 'V2'), 3)
                manifest = json.loads((self.trial / 'trial_manifest.json').read_text(encoding='utf-8'))
                self.assertTrue(any(message in error for error in manifest['validation_errors']))
    def test_eager_protocol_rejects_an_actual_graph_run(self):
        self.profile['cuda_graph_enabled'] = False
        for physics in (self.contract['physics'], self.identity['env']['physics']):
            physics.setdefault('runtime_diagnostics', {}).update(
                cuda_graph_requested=True, cuda_graph_enabled=True, cuda_graph_capture_s=0.01)
        self.save_sidecars()
        self.assertEqual(trial_checks.write_manifest(self.trial, 0, self.sha_file, 'V2'), 3)
        manifest = json.loads((self.trial / 'trial_manifest.json').read_text(encoding='utf-8'))
        self.assertIn('protocol requires eager physics execution', manifest['validation_errors'])
    def test_v2_graph_request_requires_actual_execution_evidence(self):
        self.profile['cuda_graph_enabled'] = True
        self.assertEqual(trial_checks.write_manifest(self.trial, 0, self.sha_file, 'V2'), 3)
        manifest = json.loads((self.trial / 'trial_manifest.json').read_text(encoding='utf-8'))
        self.assertIn('protocol requires actual CUDA graph execution', manifest['validation_errors'])
        for physics in (self.contract['physics'], self.identity['env']['physics']):
            physics.setdefault('runtime_diagnostics', {}).update(
                cuda_graph_requested=True, cuda_graph_enabled=True, cuda_graph_capture_s=0.01)
        self.save_sidecars()
        self.assertEqual(trial_checks.write_manifest(self.trial, 0, self.sha_file, 'V2'), 0)
    def test_frozen_body_tampering_is_rejected_even_with_old_sha_file(self):
        self.protocol.write_text(self.protocol.read_text(encoding='utf-8') + 'Changed body.\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'frozen protocol body'):
            trial_checks.write_manifest(self.trial, 0, self.sha_file, 'V2')

    def test_unfrozen_protocol_profile_is_rejected(self):
        with patch.dict(trial_checks.PROTOCOL_PROFILES, {'V2': {**self.profile, 'sha': None}}):
            with self.assertRaisesRegex(ValueError, 'not been frozen'):
                trial_checks.write_manifest(self.trial, 0, self.sha_file, 'V2')


if __name__ == '__main__':
    unittest.main()
