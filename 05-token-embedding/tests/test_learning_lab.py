"""显示与记录的行为检查；不包含课堂四个 TODO 的完整答案。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
from learning_lab import LessonLab, Progress, EVENTS, REQUIRED_MAX
from tokenizer_tools import load_tokenizer


class ProgressTests(unittest.TestCase):
    def test_restart_duplicate_and_latest_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            p = Progress(path)
            self.assertFalse(path.exists())
            self.assertTrue(p.award('observe', 'first'))
            other = Progress(path)
            self.assertFalse(other.award('observe', 'again'))
            p.award('counterexample', 'new')
            data = other.read()
            self.assertEqual(sum(EVENTS[k][1] for k in data['events']), 20)
            self.assertEqual(data['events']['observe']['first']['evidence'], 'first')
            self.assertEqual(data['events']['observe']['latest']['evidence'], 'again')
            other.reflection('文字说明待讨论')
            self.assertEqual(p.read()['reflection'], '文字说明待讨论')
            self.assertEqual(len(p.read()['events']), 2)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_corrupt_record_is_not_erased(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            path.write_text('{broken')
            with self.assertRaises(json.JSONDecodeError):
                Progress(path)
            self.assertEqual(path.read_text(), '{broken')


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'progress.json'
        self.lab = LessonLab(self.tokenizer, self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_legacy_records_still_load_and_count(self):
        self.assertIn(f'必做最高 {REQUIRED_MAX} 积分', self.lab.score.value)
        self.assertNotIn('旧版关卡', self.lab.score.value)
        self.lab.progress.award('cut_round', '旧版切一刀')
        self.lab.progress.award('counterexample', '旧版反例')
        lab = LessonLab(self.tokenizer, self.path)
        self.assertIn('另有旧版关卡留下的 25 分照常计入', lab.score.value)
        self.assertIn('<span class="metric">25</span>', lab.score.value)
        self.assertEqual(REQUIRED_MAX, 65)

    def test_row_prediction_does_not_modify_student_table(self):
        E = torch.randn(8, 4, generator=torch.Generator().manual_seed(7))
        ids = torch.tensor([0, 1, 2, 3, 0, 4, 5, 6, 7])
        original = E.clone()
        self.lab.embedding(E, ids, list('猫追狗，看星空。'))
        self.lab.prediction.value = '0'
        self.lab.row_button.click()
        self.assertNotIn('row_prediction', self.lab.progress.read()['events'])
        self.lab.prediction.value = '0,4'
        self.lab.row_button.click()
        self.assertIn('row_prediction', self.lab.progress.read()['events'])
        self.assertIn('实际变化的位置：[0, 4]', self.lab.row_result.value)
        torch.testing.assert_close(E, original)
        self.lab.position.value = 2
        self.assertIn('位置 2 的 one-hot：0 0 1 0 0 0 0 0', self.lab.embedding_view.value)
        self.lab.delta.value = 0
        self.lab.prediction.value = '无'
        self.lab.row_button.click()
        self.assertIn('增量为 0', self.lab.feedback.value)

    def test_memory_updates_and_has_no_exploration_reward(self):
        self.lab.memory()
        self.assertIn('4,915,200,000', self.lab.memory_view.value)
        self.lab.mem_controls['T'].value = 4096
        self.assertIn('9,830,400,000', self.lab.memory_view.value)
        self.assertIn('460,800,000', self.lab.memory_view.value)
        self.assertFalse(self.path.exists())

    def test_bad_implementations_fail_without_points(self):
        broken = {
            'compare': lambda parts, tok: {},
            'density': lambda tok, samples: [{'name': n, 'chars': len(t), 'tokens': 0, 'tokens_per_char': 0}
                                             for n, t in samples.items()],
            'lookup': lambda E, ids: (E[0], E[0]),
            'memory': lambda *args: {'ids': 0, 'table': 0, 'output': 0, 'one_hot': 0},
            'mask': lambda vectors, mask: vectors.mean(dim=1),
        }
        for key, fn in broken.items():
            panel = self.lab.checkpoint(key, lambda fn=fn: fn)
            panel.children[1].click()
            self.assertNotIn(key, self.lab.progress.read()['events'])
            self.assertIn('还没有完成', self.lab.feedback.value)
        self.assertFalse(self.path.exists())

    def test_dashboard_counts_core_and_shows_optional_only_when_done(self):
        self.assertIn('必做代码 0/2', self.lab.score.value)
        self.assertNotIn('选做深入', self.lab.score.value.split('<table>')[1].split('</table>')[0])
        self.lab.progress.award('lookup', 'deep dive')
        self.lab._render_progress()
        self.assertIn('必做代码 0/2', self.lab.score.value)
        self.assertIn('选做深入：两种查表方法', self.lab.score.value)

    def test_motion_off_and_reflection_is_pending(self):
        self.lab.motion.value = False
        self.lab._message('success', success=True)
        self.assertNotIn('class="win pop"', self.lab.feedback.value)
        panel = self.lab.reflection_panel()
        self.lab.notes.value = '我还不能解释为什么取行等价。'
        panel.children[2].click()
        data = self.lab.progress.read()
        self.assertEqual(data['events'], {})
        self.assertEqual(data['reflection'], self.lab.notes.value)
        self.assertIn('已保存，待讨论', self.lab.score.value)


if __name__ == '__main__':
    unittest.main()
