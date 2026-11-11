"""小游戏的行为检查：计分、揭晓、抽题与纪录都来自真实分词器。"""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
from learning_lab import Progress
from token_games import GuessGame, MATERIALS, ROUND_SIZE, BEST_KEY, judge, HIT, NEAR, MISS
from tokenizer_tools import load_tokenizer


class GameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'progress.json'
        self.progress = Progress(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def test_judge_thresholds(self):
        self.assertEqual(judge(6, 5), HIT)
        self.assertEqual(judge(22, 20), HIT)
        self.assertEqual(judge(25, 20), NEAR)
        self.assertEqual(judge(30, 20), MISS)

    def test_round_draws_five_distinct_materials_and_restart_redraws(self):
        game = GuessGame(self.tokenizer, self.progress, rng=random.Random(3))
        self.assertEqual(len(game.rounds), ROUND_SIZE)
        self.assertEqual(len(set(game.rounds)), ROUND_SIZE)
        self.assertTrue(set(game.rounds) <= set(MATERIALS))
        first = list(game.rounds)
        for _ in range(ROUND_SIZE):
            game.answer.value = '1'
            game.next_button.click()
        self.assertEqual(game.phase, 'done')
        game.restart_button.click()
        self.assertEqual((game.phase, game.index, game.score), ('guess', 0, 0))
        self.assertNotEqual(game.rounds, first)

    def test_guess_reveal_scores_and_rejects_bad_input(self):
        rounds = [('中文', '今天下午三点开会。'), ('标签', '<b>x</b>')]
        game = GuessGame(self.tokenizer, self.progress, rounds=rounds)
        game.answer.value = 'abc'
        self.assertEqual(game.phase, 'guess')
        self.assertIn('请输入一个正整数', game.view.value)
        actual = self.count(rounds[0][1])
        game.answer.value = str(actual)
        self.assertEqual(game.phase, 'revealed')
        self.assertEqual(game.score, 3)
        self.assertIn('命中', game.view.value)
        self.assertIn(f'{10_000 * actual / len(rounds[0][1]):,.0f}', game.view.value)
        game.reveal_button.click()
        self.assertEqual(game.score, 3)
        self.assertEqual(game.reveal_button.layout.display, 'none')
        game.next_button.click()
        game.answer.value = str(self.count(rounds[1][1]) * 3)
        self.assertNotIn('<b>x</b>', game.view.value)
        self.assertIn('&lt;b&gt;', game.view.value)
        game.next_button.click()
        self.assertEqual(game.phase, 'done')
        self.assertIn('class="chart"', game.view.value)
        self.assertIn('明显高估：标签', game.view.value)
        self.assertEqual(self.progress.best(BEST_KEY)['score'], 3)
        self.assertIn('guess_round', self.progress.read()['events'])

    def test_best_score_only_grows_and_keeps_existing_records(self):
        self.progress.award('compare', '已有记录')
        self.progress.award('cut_round', '旧版关卡')
        self.progress.record_best('guess', 14)
        self.progress.reflection('我的总结')
        self.assertEqual(self.progress.record_best(BEST_KEY, 12), (True, 12))
        self.assertEqual(self.progress.record_best(BEST_KEY, 8), (False, 12))
        self.assertEqual(self.progress.record_best(BEST_KEY, 15), (True, 15))
        data = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertIn('compare', data['events'])
        self.assertIn('cut_round', data['events'])
        self.assertEqual(data['games']['guess']['score'], 14)
        self.assertEqual(data['reflection'], '我的总结')


if __name__ == '__main__':
    unittest.main()
