"""Execution-profile safeguards and queued-job selection, without CUDA."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.preflight_sparsedrop_400m import (
    PARAMETERS_TOTAL, PARAMETERS_TRAINABLE, SCIENTIFIC_CONFIG, TOKENS_PER_STEP,
    selected_configuration, validate_summary,
)
from scripts.validate_sparsedrop_400m_launch import execution_profile


class ExecutionProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'cluster').mkdir()
        (self.root / 'configs/sparsedrop_400m20b').mkdir(parents=True)
        self.profiles = {str(i): {
            'batch_size': 8 if i < 4 else 32,
            'grad_accum': 4 if i < 4 else 1,
            'gradient_checkpointing': i < 4,
        } for i in range(18)}
        self.rows = []
        for index in range(18):
            row = {'index': index, 'loops': (2, 4, 6, 8, 10, 12)[index // 3],
                   'mode': ('ffn', 'attention', 'both')[index % 3],
                   'config': f'configs/sparsedrop_400m20b/{index}.json'}
            self.rows.append(row)
            cfg = {**SCIENTIFIC_CONFIG, **self.profiles[str(index)],
                   'loops': row['loops'], 'mode': row['mode']}
            (self.root / row['config']).write_text(json.dumps(cfg))
        (self.root / 'configs/sparsedrop_400m20b/manifest.json').write_text(json.dumps(self.rows))
        self.policy = self.root / 'cluster/400m_execution_profiles.json'
        self.policy.write_text(json.dumps({'profiles': self.profiles}))

    def tearDown(self):
        self.temporary.cleanup()

    def test_queued_job_selects_actual_loop_and_profile(self):
        cfg, selected = selected_configuration(root=self.root,
                                               environ={'SLURM_ARRAY_TASK_ID': '10'})
        self.assertEqual((cfg['loops'], cfg['mode']), (8, 'attention'))
        self.assertEqual((cfg['batch_size'], cfg['grad_accum']), (32, 1))
        self.assertFalse(cfg['gradient_checkpointing'])
        self.assertEqual(selected['index'], 10)
        self.assertEqual(cfg['batch_size'] * cfg['grad_accum'] * cfg['seq_len'], TOKENS_PER_STEP)
        self.assertEqual(cfg['token_budget'], 20_000_000_000)

    def test_active_legacy_profile_and_standalone_default_are_preserved(self):
        cfg, _ = selected_configuration(root=self.root, environ={'SLURM_ARRAY_TASK_ID': '1'})
        self.assertEqual((cfg['batch_size'], cfg['grad_accum']), (8, 4))
        self.assertTrue(cfg['gradient_checkpointing'])
        cfg, selected = selected_configuration(root=self.root, environ={})
        self.assertEqual(cfg, SCIENTIFIC_CONFIG)
        self.assertEqual(selected['selection'], 'standalone_h12_default')

    def test_explicit_config_is_registered_and_cannot_change_array_index(self):
        path = self.root / self.rows[17]['config']
        cfg, selected = selected_configuration(path, root=self.root, environ={})
        self.assertEqual((cfg['loops'], selected['index']), (12, 17))
        with self.assertRaisesRegex(RuntimeError, 'Slurm array index'):
            selected_configuration(path, root=self.root, environ={'SLURM_ARRAY_TASK_ID': '1'})
        unregistered = self.root / 'arbitrary.json'
        unregistered.write_text(path.read_text())
        with self.assertRaisesRegex(RuntimeError, 'registered'):
            selected_configuration(unregistered, root=self.root, environ={})

    def test_equal_effective_batch_cannot_bypass_specific_profile(self):
        path = self.root / self.rows[17]['config']
        cfg = json.loads(path.read_text())
        cfg.update(batch_size=16, grad_accum=2)
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(RuntimeError, 'approved execution profile'):
            selected_configuration(path, root=self.root, environ={})

    def test_profile_policy_requires_complete_exact_and_valid_profiles(self):
        mutations = (
            lambda p: p.pop('0'),
            lambda p: p['17'].update(grad_accum=2),
            lambda p: p['17'].update(grad_accum=True),
            lambda p: p['17'].update(gradient_checkpointing=0),
            lambda p: p['17'].update(unapproved_option=True),
        )
        for mutate in mutations:
            profiles = deepcopy(self.profiles)
            mutate(profiles)
            self.policy.write_text(json.dumps({'profiles': profiles}))
            with self.assertRaises(ValueError):
                execution_profile(17, self.root)

    def test_execution_change_cannot_change_learning_rate(self):
        path = self.root / self.rows[17]['config']
        cfg = json.loads(path.read_text())
        cfg['lr'] = 0.001
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(RuntimeError, 'scientific contract'):
            selected_configuration(path, root=self.root, environ={})

    def test_diagnostics_validate_selected_depth_and_reject_missing_row(self):
        cfg, _ = selected_configuration(root=self.root, environ={'SLURM_ARRAY_TASK_ID': '7'})
        loops, rows = cfg['loops'], cfg['loops'] * 6 * 2
        output = self.root / 'synthetic'
        output.mkdir()
        diagnostics = {
            'protocol': {'sampled_tokens': 512, 'sample_position_sha256': 'fixed'},
            'cross_loop': {name: [[1.0] * loops for _ in range(loops)]
                           for name in ('state_cosine', 'update_cosine', 'linear_cka')},
        }
        summary = {'status': 'complete', 'step': 3, 'tokens_seen': 3 * TOKENS_PER_STEP,
                   'compute_tokens_seen': 3 * TOKENS_PER_STEP,
                   'completion_reason': 'stop_after_steps', 'token_budget_completed': False,
                   'original_token_budget': 20_000_000_000, 'lr_schedule_max_steps': 1_220_704,
                   'final_full_validation': {'loss': 1.0, 'perplexity': 2.718,
                                             'diagnostics': diagnostics},
                   'peak_gpu_memory_bytes': 12345, 'train_seconds': 1.0}
        metadata = {'parameters_total': PARAMETERS_TOTAL,
                    'parameters_trainable': PARAMETERS_TRAINABLE, 'arguments': cfg,
                    'restored_step': 0, 'initial_parameters_sha256': 'initial'}
        (output / 'summary.json').write_text(json.dumps(summary))
        (output / 'config.json').write_text(json.dumps(metadata))
        (output / 'sparsity_final.json').write_text(json.dumps({
            'coverage': {'observed_rows': rows}, 'step': 3}))

        def validate(actual, **kwargs):
            self.assertEqual(kwargs, {'loops': loops, 'num_layers': 6, 'hidden_size': 2048})
            self.assertEqual(actual, diagnostics)
            return [{}] * rows

        result = validate_summary(output, 3, validate, cfg)
        self.assertEqual(result['diagnostic_rows'], 72)
        self.assertEqual(result['loop_matrix_dimensions'], [6, 6])
        with self.assertRaisesRegex(RuntimeError, '72 branches'):
            validate_summary(output, 3, lambda *a, **kw: [{}] * (rows - 1), cfg)


if __name__ == '__main__':
    unittest.main()
