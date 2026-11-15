import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_monitor import TrainingMonitor
from expand_industry import clean,canonical

class RunMetadataTests(unittest.TestCase):
    def test_completed_and_failed_runs_keep_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            with TrainingMonitor(p/'ok/tensorboard',{'label':'V100 expanded','note':'continue own model'}):pass
            r=json.loads((p/'ok/run.json').read_text())
            self.assertEqual(r['status'],'completed');self.assertEqual(r['label'],'V100 expanded')
            with self.assertRaises(RuntimeError):
                with TrainingMonitor(p/'bad/tensorboard',{'label':'probe'}):raise RuntimeError('test failure')
            r=json.loads((p/'bad/run.json').read_text());self.assertEqual(r['status'],'failed')
            self.assertEqual(r['error_type'],'RuntimeError')

    def test_quality_filters_and_canonical_duplicates(self):
        self.assertEqual(canonical('机器 学习，AI。'),canonical('机器学习,ai'))
        self.assertIsNone(clean('a'*300))
        self.assertIsNone(clean('这是反复出现的模板句子缺少新的信息。'*30))
        text=''.join(f'第{i}段讨论机器学习如何处理不同领域的数据以及模型的预测结果。' for i in range(15))
        self.assertEqual(clean(text),text)
        self.assertIsNone(clean(text+'\ufffd'))

if __name__=='__main__':unittest.main()
