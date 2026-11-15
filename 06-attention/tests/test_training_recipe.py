"""Training-entry regressions for recipe changes and WSD checkpoint continuation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indexed_tokens import TokenWindowWriter
from lm_components import learning_rate_at_step
from train_lm import main, restore


SCRIPT = Path(__file__).resolve().parents[1] / 'train_lm.py'


class TrainingRecipeTests(unittest.TestCase):
    def test_wsd_warmup_plateau_and_cooldown_boundaries(self):
        rates = [learning_rate_at_step(step, 10, 2, .01, .1,
                                       schedule='wsd', cooldown_steps=3)
                 for step in range(1, 11)]
        self.assertEqual(rates[0], .005)
        self.assertEqual(rates[1:7], [.01] * 6)
        self.assertAlmostEqual(rates[7], .00775)
        self.assertAlmostEqual(rates[8], .00325)
        self.assertAlmostEqual(rates[9], .001)
        self.assertEqual(learning_rate_at_step(1, 10, 2, .01, schedule='constant'), .01)
        with self.assertRaisesRegex(ValueError, 'cooldown'):
            learning_rate_at_step(4, 10, 3, .01, schedule='wsd', cooldown_steps=8)

    def test_cli_recipe_resume_and_explicit_optimizer_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            vocabulary = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, 'a': 3, 'b': 4, 'c': 5}
            tokenizer = Tokenizer(models.WordLevel(vocabulary))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(data / 'tokenizer.json'))
            writer = TokenWindowWriter(data, context_length=4)
            for text in ('a b c a b c', 'b c a b c a', 'c a b c a b'):
                writer.add_document([1, *tokenizer.encode(text).ids, 2])
            writer.finish()
            (data / 'validation.json').write_text(json.dumps(['a c b a c b', 'b a c b a c']))
            command = [sys.executable, str(SCRIPT), 'train', '--data-dir', str(data),
                       '--epochs', '2', '--batch-size', '2', '--context-length', '4',
                       '--d-model', '8', '--num-layers', '1', '--num-heads', '2',
                       '--eval-every', '2', '--train-probe-windows', '2', '--log-every', '1',
                       '--no-full-train-eval', '--observation-prompts', 'a',
                       '--learning-rate', '.003', '--lr-schedule', 'wsd',
                       '--warmup-ratio', '.2', '--cooldown-ratio', '.5']
            if os.environ.get('RESUME_TEST_DEVICE') == 'cuda':
                command += ['--device', 'cuda', '--precision', 'fp16']

            def run(name, *options, error=None):
                result = subprocess.run(command + ['--run-dir', str(root / name), *options],
                                        capture_output=True, text=True)
                if error is not None:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stdout + result.stderr)
                    return
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return torch.load(root / name / 'last.pt', map_location='cpu', weights_only=True)

            recipe = ['--optimizer', 'adamw', '--beta2', '.95', '--weight-decay', '.1', '--dropout', '0']
            full = run('full', *recipe)
            run('partial', *recipe, '--stop-after-step', '2')
            resumed = run('resumed', '--resume-checkpoint', str(root / 'partial/last.pt'))
            self.assertEqual(full['checks'], resumed['checks'])
            self.assertEqual(full['tokens_seen'], 42)
            self.assertEqual(resumed['tokens_seen'], 42)
            for name, weight in full['model'].items():
                torch.testing.assert_close(weight, resumed['model'][name], rtol=0, atol=0)
            self.assertEqual(full['training_config']['cooldown_steps'], 3)
            self.assertEqual(full['training_config']['optimizer'], 'adamw')
            self.assertEqual(resumed['training_config']['betas'], [.9, .95])
            for state, other in zip(full['optimizer']['state'].values(), resumed['optimizer']['state'].values()):
                for key, value in state.items():
                    torch.testing.assert_close(value, other[key], rtol=0, atol=0)

            origin = str(root / 'full/last.pt')
            arguments = command[1:] + ['--run-dir', str(root / 'inspect-live-dropout'),
                                       '--init-checkpoint', origin, '--dropout', '.23']
            with patch.object(sys, 'argv', arguments), patch('training_loop.train') as train:
                main()
            live_model = train.call_args.args[1]
            self.assertEqual(live_model.config['dropout'], .23)
            self.assertTrue(all(module.p == .23 for module in live_model.modules()
                                if isinstance(module, torch.nn.Dropout)))
            continued = run('continued', '--init-checkpoint', origin,
                            '--keep-optimizer-state', '--dropout', '.1')
            self.assertEqual(continued['training_config']['optimizer'], 'adamw')
            self.assertEqual(continued['optimizer']['param_groups'][0]['betas'], (.9, .95))
            self.assertEqual(continued['optimizer']['param_groups'][0]['weight_decay'], .1)
            self.assertEqual({int(state['step']) for state in continued['optimizer']['state'].values()}, {12})
            model, _, _ = restore(root / 'continued/last.pt', 'cpu')
            self.assertEqual(model.config['dropout'], .1)
            self.assertTrue(all(module.p == .1 for module in model.modules() if isinstance(module, torch.nn.Dropout)))

            reset = run('reset', '--init-checkpoint', origin, '--optimizer', 'adam', '--dropout', '.1')
            self.assertEqual(reset['training_config']['betas'], [.9, .999])
            self.assertEqual(reset['optimizer']['param_groups'][0]['weight_decay'], 0.)
            self.assertEqual({int(state['step']) for state in reset['optimizer']['state'].values()}, {6})
            for key, value in (('--optimizer', 'adam'), ('--beta2', '.999'), ('--weight-decay', '0')):
                run('bad-' + key[2:], '--init-checkpoint', origin, '--keep-optimizer-state',
                    key, value, error='复用优化器状态')
            run('bad-dropout', '--resume-checkpoint', str(root / 'partial/last.pt'),
                '--dropout', '.1', error='相同dropout')
            run('bad-cooldown', '--resume-checkpoint', str(root / 'partial/last.pt'),
                '--cooldown-ratio', '.2', error='cooldown_steps')


if __name__ == '__main__':
    unittest.main()
