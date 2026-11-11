"""真实加载事件与张量显示检查；不提供课堂 TODO 的答案。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
import tokenizer_tools
from lab_visuals import TokenizerLoadPanel, TensorExplorer, STAGES


class LoaderTests(unittest.TestCase):
    def test_real_file_events_and_structure(self):
        panel = TokenizerLoadPanel()
        tokenizer = panel.load()
        self.assertEqual([e['stage'] for e in panel.events if e['status'] == 'done'], list(STAGES))
        self.assertEqual(panel.latest['read']['size_bytes'], (COURSE/'assets/tokenizer.json').stat().st_size)
        self.assertEqual(panel.latest['verify']['sha256'], tokenizer_tools.TOKENIZER_SHA256)
        self.assertEqual(panel.latest['parse']['model_type'], 'BPE')
        self.assertGreater(panel.latest['parse']['merge_rules'], 0)
        self.assertEqual(panel.latest['probe']['vocab_size'], tokenizer.get_vocab_size(with_added_tokens=True))
        self.assertEqual(panel.latest['probe']['restored'], panel.latest['probe']['sample'])
        self.assertIn('实际试编码通过', panel.summary.value)

    def test_missing_and_corrupt_files_stop_at_real_stage(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tokenizer_tools, 'ASSET_DIR', Path(directory)):
            panel = TokenizerLoadPanel()
            with self.assertRaises(FileNotFoundError):
                panel.load()
            self.assertEqual(panel.latest['locate']['status'], 'error')
            self.assertNotIn('read', panel.latest)
            (Path(directory)/'tokenizer.json').write_text('{}')
            with self.assertRaises(ValueError):
                panel.load()
            self.assertEqual(panel.latest['verify']['status'], 'error')
            self.assertNotIn('parse', panel.latest)
            self.assertNotIn('probe', panel.latest)
            self.assertNotIn('实际试编码通过', panel.summary.value)
            self.assertEqual(panel.steps.selected_index, 2)
            raw = (COURSE/'assets/tokenizer.json').read_bytes()
            # 保持大小不变，仅改一个字节，仍须被指纹校验挡住。
            (Path(directory)/'tokenizer.json').write_bytes(b'!' + raw[1:])
            with self.assertRaises(ValueError):
                panel.load()
            self.assertEqual(panel.latest['read']['size_bytes'], len(raw))
            self.assertEqual(panel.latest['verify']['status'], 'error')
            self.assertNotIn('parse', panel.latest)

    def test_retry_after_success_clears_stale_ready_details(self):
        panel = TokenizerLoadPanel()
        panel.load()
        with tempfile.TemporaryDirectory() as directory, patch.object(tokenizer_tools, 'ASSET_DIR', Path(directory)):
            with self.assertRaises(FileNotFoundError):
                panel.load()
        self.assertNotIn('probe', panel.latest)
        self.assertIn('尚未执行', panel.details[-1].value)
        self.assertNotIn('实际试编码通过', panel.summary.value)


class TensorTests(unittest.TestCase):
    def test_index_slice_shapes_and_axes_without_mutation(self):
        E = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        original = E.clone()
        viewer = TensorExplorer({'E': (E, ['词表项 V', '向量维度 D'])})
        viewer.row.value = 1
        viewer.mode.value = 'index'
        result, axes, expression = viewer.selection()
        self.assertEqual(result.shape, (4,))
        self.assertEqual(axes, ['向量维度 D'])
        self.assertEqual(expression, 'E[1]')
        torch.testing.assert_close(result, E[1])
        viewer.mode.value = 'slice'
        result, axes, expression = viewer.selection()
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(axes, ['词表项 V', '向量维度 D'])
        self.assertEqual(expression, 'E[1:2]')
        torch.testing.assert_close(E, original)
        self.assertIn('12 × 4 = 48 bytes', viewer.view.value)

    def test_vector_scalar_and_three_dimensional_batch(self):
        ids = torch.tensor([3, 1, 3])
        batch = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
        viewer = TensorExplorer({'ids': (ids, ['位置 T']), 'batch': (batch, ['批次 B', '位置 T', '维度 D'])})
        viewer.mode.value = 'index'
        self.assertEqual(viewer.selection()[0].shape, ())
        self.assertEqual(viewer.selection()[1], [])
        viewer.mode.value = 'slice'
        self.assertEqual(viewer.selection()[0].shape, (1,))
        viewer.source.value = 'batch'
        viewer.batch.value = 1
        result, axes, expression = viewer.selection()
        self.assertEqual(result.shape, (5, 4))
        torch.testing.assert_close(result, batch[1])
        self.assertEqual(axes, ['位置 T', '维度 D'])
        viewer.mode.value = 'index'
        viewer.row.value = 4
        torch.testing.assert_close(viewer.selection()[0], batch[1, 4])
        viewer.source.value = 'ids'
        self.assertEqual(viewer.batch.layout.display, 'none')
        self.assertEqual(viewer.mode.value, 'whole')

    def test_large_display_is_bounded_and_labels_are_escaped(self):
        viewer = TensorExplorer({'<script>': (torch.zeros(100, 20), ['行<script>', '列'])})
        self.assertEqual(viewer.view.value.count('<tr>'), 17)
        self.assertEqual(viewer.view.value.count('<td '), 16*12)
        self.assertNotIn('<script>', viewer.view.value)
        self.assertIn('(100, 20)', viewer.view.value)
        self.assertIn('8,000 bytes', viewer.view.value)
        with self.assertRaises(ValueError):
            TensorExplorer({'bad': (torch.zeros(3, 4), ['只有一个轴'])})


if __name__ == '__main__':
    unittest.main()
