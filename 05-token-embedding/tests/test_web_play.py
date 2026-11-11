"""网页试玩：测量、一轮猜数和本机页面。进度写入临时文件。"""
import json
import random
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
from learning_lab import Progress
from tokenizer_tools import load_tokenizer
from web.engine import PlayEngine
from web.server import make_handler


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'progress.json'
        self.progress = Progress(self.path)
        self.engine = PlayEngine(self.tokenizer, self.progress, random.Random(1))

    def tearDown(self):
        self.directory.cleanup()

    def test_measure_real_tokenizer_and_reject_a_whole_contract(self):
        result = self.engine.measure_text('你好，世界！')
        self.assertEqual(result['tokens'], 4)
        self.assertEqual(result['chars'], 6)
        self.assertAlmostEqual(result['per_10k_chars'], 10_000 * 4 / 6)
        self.assertEqual(result['blocks'][0]['label'], '你好')
        notes = '结论：\n1. 周五上线\n2. 张三负责回归测试'
        labels = [block['label'] for block in self.engine.measure_text(notes)['blocks']]
        self.assertIn('周 · 2/3', labels)
        self.assertIn('五', labels)
        empty = self.engine.measure_text('')
        self.assertEqual(empty['tokens'], 0)
        self.assertIsNone(empty['tokens_per_char'])
        with self.assertRaises(ValueError):
            self.engine.measure_text('字' * 4001)

    def test_round_hides_answer_until_reveal_and_keeps_other_records(self):
        self.progress.award('compare', '已有记录')
        started = self.engine.start()
        self.assertNotIn('tokens', started)
        self.assertEqual(started['total'], 5)
        with self.assertRaises(ValueError):
            self.engine.reveal(started['id'], 'abc')
        actual = len(self.tokenizer.encode(started['text'], add_special_tokens=False).ids)
        revealed = self.engine.reveal(started['id'], actual)
        self.assertEqual(revealed['verdict'], 'hit')
        self.assertEqual(revealed['score'], 3)
        self.assertEqual(len(revealed['blocks']), actual)
        for _ in range(4):
            self.assertEqual(self.engine.advance(started['id'])['phase'], 'guess')
            self.engine.reveal(started['id'], 1)
        step = self.engine.advance(started['id'])
        self.assertEqual(step['phase'], 'done')
        self.assertEqual(len(step['rows']), 5)
        self.assertTrue(any('费' in line for line in step['lines']))
        data = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(data['events']['compare']['latest']['evidence'], '已有记录')
        self.assertEqual(data['games']['guess5']['score'], 3)
        self.assertIn('guess_round', data['events'])


class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.directory = directory
        progress = Progress(Path(directory.name) / 'progress.json')
        engine = PlayEngine(load_tokenizer(), progress, random.Random(2))
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(engine))
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.httpd.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.thread.join(timeout=3)
        cls.directory.cleanup()

    def test_page_and_measure_round_trip(self):
        page = urlopen(self.base, timeout=5).read().decode()
        self.assertIn('打一段文字，马上看见它被切成多少 token', page)
        request = Request(self.base + '/api/measure', data=json.dumps({'text': '<b>合同</b>'}).encode(),
                          headers={'Content-Type': 'application/json'})
        payload = json.loads(urlopen(request, timeout=5).read().decode())
        self.assertGreater(payload['tokens'], 0)
        self.assertTrue(all(isinstance(block['label'], str) for block in payload['blocks']))


if __name__ == '__main__':
    unittest.main()
