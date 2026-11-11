"""网页试玩的计分。分词和判定都走课程里的同一套函数，不另写一套规则。"""
import secrets

from token_games import MATERIALS, ROUND_SIZE, BEST_KEY, POINTS, measure, judge, round_report

MAX_CHARS = 4000
WORDS = {'hit': '命中！', 'near': '接近', 'miss': '偏得有点远'}


class PlayEngine:
    def __init__(self, tokenizer, progress, rng):
        self.tokenizer = tokenizer
        self.progress = progress
        self.rng = rng
        self.sessions = {}

    def measure_text(self, text):
        if not isinstance(text, str):
            raise ValueError('需要一段文字。')
        if len(text) > MAX_CHARS:
            raise ValueError(f'先控制在 {MAX_CHARS} 个字符以内。这一页用来看切分，不是用来跑整份合同。')
        return measure(self.tokenizer, text)

    def start(self):
        session_id = secrets.token_hex(8)
        self.sessions[session_id] = {
            'rounds': self.rng.sample(list(MATERIALS), ROUND_SIZE),
            'index': 0, 'score': 0, 'streak': 0, 'best_streak': 0,
            'history': [], 'phase': 'guess',
        }
        self._trim()
        return {'id': session_id, **self._question(self.sessions[session_id])}

    def reveal(self, session_id, raw_guess):
        session = self._session(session_id)
        if session['phase'] != 'guess':
            raise ValueError('这一题已经揭晓了。')
        try:
            guess = int(str(raw_guess).strip())
        except (TypeError, ValueError):
            guess = 0
        if guess <= 0:
            raise ValueError('请输入一个正整数。')
        category, text = session['rounds'][session['index']]
        measured = measure(self.tokenizer, text)
        verdict = judge(guess, measured['tokens'])
        session['score'] += POINTS[verdict]
        session['streak'] = session['streak'] + 1 if verdict == 'hit' else 0
        session['best_streak'] = max(session['best_streak'], session['streak'])
        session['history'].append((category, measured['chars'], guess, measured['tokens'], verdict))
        fresh = self.progress.award('observe', f'猜 token 数：{category} 真实 {measured["tokens"]}，猜 {guess}。')
        session['phase'] = 'revealed'
        return {
            'id': session_id, 'phase': 'revealed', 'verdict': verdict, 'words': WORDS[verdict],
            'guess': guess, 'score': session['score'], 'streak': session['streak'],
            'fresh_observe': fresh, 'last': session['index'] == len(session['rounds']) - 1,
            'best': self._best(), 'full': 3 * len(session['rounds']), **measured,
        }

    def advance(self, session_id):
        session = self._session(session_id)
        if session['phase'] != 'revealed':
            raise ValueError('先揭晓这一题。')
        session['index'] += 1
        if session['index'] < len(session['rounds']):
            session['phase'] = 'guess'
            return {'id': session_id, **self._question(session)}
        session['index'] -= 1
        session['phase'] = 'done'
        report = round_report(session['history'])
        new_best, best = self.progress.record_best(BEST_KEY, session['score'])
        fresh = self.progress.award(
            'guess_round',
            f'猜 token 数一轮 {session["score"]}/{3 * ROUND_SIZE}，命中 {report["hits"]} 题。')
        return {
            'id': session_id, 'phase': 'done', 'score': session['score'], 'full': 3 * ROUND_SIZE,
            'hits': report['hits'], 'best_streak': session['best_streak'], 'lines': report['lines'],
            'rows': report['rows'], 'new_best': new_best, 'best': best, 'fresh_round': fresh,
        }

    def _question(self, session):
        category, text = session['rounds'][session['index']]
        return {
            'phase': 'guess', 'index': session['index'], 'total': len(session['rounds']),
            'category': category, 'text': text, 'chars': len(text),
            'score': session['score'], 'best': self._best(), 'full': 3 * len(session['rounds']),
        }

    def _best(self):
        record = self.progress.best(BEST_KEY)
        return record['score'] if record else None

    def _session(self, session_id):
        try:
            return self.sessions[session_id]
        except KeyError:
            raise ValueError('这一轮已经找不到了。再来一轮即可。') from None

    def _trim(self):
        while len(self.sessions) > 20:
            del self.sessions[next(iter(self.sessions))]
