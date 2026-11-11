"""课程 5 的小游戏「猜 token 数」：每轮 5 题，题目来自方案里常见的材料。成绩写入本地进度。"""
from html import escape
import random

import ipywidgets as w

from theme import frame

# 方案里常见的材料。每轮随机抽 5 段，再来一轮会换题。
MATERIALS = [
    ('合同条款', '乙方应于每季度结束后十个工作日内提交服务报告。'),
    ('客服 FAQ', '问：订单发货后多久能到？答：一般 3 到 5 个工作日。'),
    ('会议纪要', '结论：\n1. 周五上线\n2. 张三负责回归测试'),
    ('中英混合', '请帮我review一下这个PR'),
    ('英文产品手册', 'Press and hold the power button for ten seconds to reset the device.'),
    ('英文邮件', 'Please summarize the attached contract in three bullet points.'),
    ('工具返回的 JSON', '{"order_id": "A1029", "status": "shipped", "eta_days": 3}'),
    ('代码', 'for i in range(10): print(i)'),
    ('日期时间', '2026-10-31 21:45'),
    ('聊天里的 emoji', '好耶🎉🎉🎉'),
]
ROUND_SIZE = 5
BEST_KEY = 'guess5'
HIT, NEAR, MISS = 'hit', 'near', 'miss'
POINTS = {HIT: 3, NEAR: 1, MISS: 0}
HUES = [210, 30, 140, 280, 0, 180, 60, 320, 100, 250]
COLORS = {HIT: '#2da44e', NEAR: '#d29922', MISS: '#e5534b'}


def visible(text):
    return text.replace(' ', '␣').replace('\n', '↵')


def token_label(tokenizer, token_id):
    piece = tokenizer.decode([token_id], skip_special_tokens=False)
    return visible(piece) if piece and '\ufffd' not in piece else '字节片段'


def _block_labels(tokenizer, text, ids, offsets):
    """完整的 token 用解码结果；一个字被拆成几段字节时，标成「周 · 2/3」。"""
    pieces = []
    for token_id, (start, end) in zip(ids, offsets):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        whole = bool(decoded) and '\ufffd' not in decoded
        pieces.append((token_id, whole, visible(decoded if whole else text[start:end])))
    labels = []
    index = 0
    while index < len(pieces):
        token_id, whole, shown = pieces[index]
        if whole:
            labels.append((token_id, shown))
            index += 1
            continue
        end = index
        while end < len(pieces) and not pieces[end][1]:
            end += 1
        count = end - index
        for number, (token_id, _, shown) in enumerate(pieces[index:end], start=1):
            text_label = shown or '字节'
            labels.append((token_id, text_label if count == 1 else f'{text_label} · {number}/{count}'))
        index = end
    return labels


def token_strip(tokenizer, text, animate):
    """颜色只由 token 编号决定：同一个 token 在任何地方都是同一种颜色。"""
    blocks = []
    for i, block in enumerate(measure(tokenizer, text)['blocks']):
        style = f'background:hsla({block["hue"]},70%,55%,.16);border-color:hsla({block["hue"]},60%,50%,.55)'
        if animate:
            style += f';animation-delay:{i * 70}ms'
        blocks.append(f'<span class="tok{" in" if animate else ""}" style="{style}" title="ID {block["id"]}">'
                      f'{escape(block["label"])}</span>')
    return '<div class="toks">' + ''.join(blocks) + '</div>'


def measure(tokenizer, text):
    """返回这段文字的 token 块。颜色只由编号决定，空文本不计算比例。"""
    encoding = tokenizer.encode(text, add_special_tokens=False)
    blocks = [{'id': token_id, 'label': label, 'hue': HUES[token_id % len(HUES)]}
              for token_id, label in _block_labels(tokenizer, text, encoding.ids, encoding.offsets)]
    chars, count = len(text), len(encoding.ids)
    ratio = count / chars if chars else None
    return {'chars': chars, 'tokens': count, 'tokens_per_char': ratio,
            'per_10k_chars': None if ratio is None else 10_000 * ratio, 'blocks': blocks}


def round_report(history):
    """history 每项是 (类型, 字符数, 猜测, 真实, 判定)。"""
    hits = sum(verdict == HIT for *_, verdict in history)
    over = [category for category, _, guess, actual, verdict in history if verdict == MISS and guess > actual]
    under = [category for category, _, guess, actual, verdict in history if verdict == MISS and guess < actual]
    dense = max(history, key=lambda row: row[3] / row[1])
    sparse = min(history, key=lambda row: row[3] / row[1])
    lines = []
    if over:
        lines.append('明显高估：' + '、'.join(over))
    if under:
        lines.append('明显低估：' + '、'.join(under))
    lines.append(f'这一轮最“费” token 的是{dense[0]}（每字符 {dense[3] / dense[1]:.2f}），'
                 f'最省的是{sparse[0]}（每字符 {sparse[3] / sparse[1]:.2f}），差 {dense[3] / dense[1] / (sparse[3] / sparse[1]):.1f} 倍。')
    rows = [{'category': category, 'chars': chars, 'guess': guess, 'actual': actual,
             'ratio': actual / chars, 'verdict': verdict}
            for category, chars, guess, actual, verdict in history]
    return {'hits': hits, 'lines': lines, 'rows': rows}


def judge(guess, actual):
    error = abs(guess - actual)
    if error <= 1 or error <= actual * 0.10:
        return HIT
    return NEAR if error <= actual * 0.30 else MISS


class GuessGame:
    def __init__(self, tokenizer, progress, motion=lambda: True, rounds=None, rng=None):
        self.tokenizer, self.progress, self.motion = tokenizer, progress, motion
        self.fixed_rounds, self.rng = rounds, rng or random.Random()
        self.answer = w.Text(placeholder='你猜几个 token？输入后回车', continuous_update=False,
                             layout=w.Layout(width='260px'))
        self.reveal_button = w.Button(description='揭晓', button_style='primary')
        self.next_button = w.Button(description='下一题')
        self.restart_button = w.Button(description='再来一轮（换题）', button_style='primary')
        self.view = w.HTML()
        self.answer.observe(lambda change: self.reveal(), names='value')
        self.reveal_button.on_click(lambda _: self.reveal())
        self.next_button.on_click(lambda _: self.next())
        self.restart_button.on_click(lambda _: self.start())
        self.widget = w.VBox([self.view, w.HBox([self.answer, self.reveal_button,
                                                 self.next_button, self.restart_button])])
        self.start()

    def start(self):
        self.rounds = list(self.fixed_rounds or self.rng.sample(MATERIALS, ROUND_SIZE))
        self.index, self.score, self.streak, self.best_streak = 0, 0, 0, 0
        self.history = []
        self._ask()

    @property
    def full_score(self):
        return 3 * len(self.rounds)

    def _controls(self, phase):
        self.phase = phase
        self.answer.disabled = phase != 'guess'
        self.reveal_button.layout.display = '' if phase == 'guess' else 'none'
        self.next_button.layout.display = '' if phase == 'revealed' else 'none'
        self.restart_button.layout.display = '' if phase == 'done' else 'none'
        self.answer.layout.display = 'none' if phase == 'done' else ''

    def _header(self):
        best = self.progress.best(BEST_KEY)
        streak = (f'　<span class="pop hit">连击 ×{self.streak}</span>'
                  if self.streak >= 2 and self.phase == 'revealed' else '')
        return (f'<div class="head"><b>第 {self.index + 1} / {len(self.rounds)} 题 · {escape(self.rounds[self.index][0])}</b>'
                f'<span>得分 <b>{self.score}</b>{streak}　<span class="muted">最好成绩 '
                f'{best["score"] if best else "—"} / {self.full_score}</span></span></div>')

    def _ask(self, notice=''):
        self._controls('guess')
        self.answer.value = ''
        category, text = self.rounds[self.index]
        self.view.value = frame(self._header() + f'<div class="q">{escape(text)}</div>'
            f'<p class="muted">这段有 {len(text)} 个字符。先凭感觉报一个 token 数，误差在 1 个或 10% 以内算命中。</p>'
            + (f'<p class="miss">{escape(notice)}</p>' if notice else ''))

    def reveal(self):
        if self.phase != 'guess' or not self.answer.value.strip():
            return
        try:
            guess = int(self.answer.value.strip())
        except ValueError:
            guess = 0
        if guess <= 0:
            self._ask('请输入一个正整数。')
            return
        category, text = self.rounds[self.index]
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        verdict = judge(guess, len(ids))
        self.score += POINTS[verdict]
        self.streak = self.streak + 1 if verdict == HIT else 0
        self.best_streak = max(self.best_streak, self.streak)
        self.history.append((category, len(text), guess, len(ids), verdict))
        fresh = self.progress.award('observe', f'猜 token 数：{category} 真实 {len(ids)}，猜 {guess}。')
        self._controls('revealed')
        ratio = len(ids) / len(text)
        words = {HIT: '命中！', NEAR: '接近', MISS: '偏得有点远'}[verdict]
        self.view.value = frame(self._header() + f'<div class="q">{escape(text)}</div>'
            + token_strip(self.tokenizer, text, self.motion())
            + self._ruler(guess, len(ids), verdict)
            + f'<p><span class="big {verdict}">{words}</span>　真实 {len(ids)} 个 token，你猜 {guess}。'
            + ('　<span class="muted">+5 探索积分</span>' if fresh else '') + '</p>'
            + f'<p>每个字符约 <b>{ratio:.2f}</b> 个 token；照这个比例，1 万个字符的同类材料约 '
            f'<b>{10_000 * ratio:,.0f}</b> 个 token。<span class="muted">只是这一小段、这一个分词器的比例。</span></p>'
            + '<p class="muted">␣ 是空格，↵ 是换行；悬停可看编号。</p>')
        self.next_button.description = '看总成绩' if self.index == len(self.rounds) - 1 else '下一题'

    def _ruler(self, guess, actual, verdict):
        top = max(guess, actual) * 1.25 + 1
        at = lambda n: n / top * 100
        low, high = sorted((at(guess), at(actual)))
        color = COLORS[verdict]
        return ('<div class="ruler"><div class="line"></div>'
                f'<div class="gap" style="left:{low}%;width:{high - low}%;background:{color}"></div>'
                f'<div class="mark up" style="left:{at(guess)}%;color:{color}">你猜 {guess}<i></i></div>'
                f'<div class="mark down" style="left:{at(actual)}%"><i></i>真实 {actual}</div>'
                '<div class="mark down muted" style="left:0"><i></i>0</div>'
                '</div>')

    def _chart(self):
        top = max(max(a, g) / n for _, n, g, a, _ in self.history) * 1.1
        rows = []
        for category, n, guess, actual, verdict in self.history:
            rows.append(f'<span>{escape(category)}</span><div class="track">'
                        f'<div class="bar" style="width:{actual / n / top * 100}%;background:{COLORS[verdict]}"></div>'
                        f'<div class="tick" style="left:{guess / n / top * 100}%"></div></div>'
                        f'<span class="muted">{actual / n:.2f}</span>')
        return ('<p class="muted" style="margin-top:14px">每字符多少 token：色条是真实值（颜色＝命中、接近、偏远），'
                '竖线是你猜的。</p><div class="chart">' + ''.join(rows) + '</div>')

    def next(self):
        if self.phase != 'revealed':
            return
        self.index += 1
        if self.index < len(self.rounds):
            self._ask()
        else:
            self.index -= 1
            self._finish()

    def _finish(self):
        self._controls('done')
        new_best, best = self.progress.record_best(BEST_KEY, self.score)
        report = round_report(self.history)
        hits, lines = report['hits'], report['lines']
        fresh = self.progress.award('guess_round', f'猜 token 数一轮 {self.score}/{self.full_score}，命中 {hits} 题。')
        record = (f'<span class="pop hit big">新纪录！</span>' if new_best else f'<span class="muted">最好成绩 {best}</span>')
        self.view.value = frame(f'<div class="head"><b>一轮结束</b>{record}</div>'
            f'<p><span class="big">{self.score}</span> / {self.full_score} 分　命中 {hits} 题　最长连击 {self.best_streak}'
            + ('　<span class="muted">+10 探索积分</span>' if fresh else '') + '</p>'
            + ''.join(f'<p>{escape(line)}</p>' for line in lines) + self._chart()
            + '<p class="muted" style="margin-top:12px">同样多的字符，token 数可以差好几倍。所以给客户估算时，要按材料类型分别抽样测比例，'
            '不能拿一个比例套所有材料。第 4 节你会把这个测量写成函数，第 6 节用它估一份合同。</p>')


class TokenGames:
    def __init__(self, tokenizer, progress, motion=lambda: True):
        self.tokenizer, self.progress, self.motion = tokenizer, progress, motion

    def guess(self):
        self.guess_game = GuessGame(self.tokenizer, self.progress, self.motion)
        return self.guess_game.widget
