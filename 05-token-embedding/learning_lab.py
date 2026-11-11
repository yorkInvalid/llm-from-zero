"""课程 5 的显示、检查与本地进度。核心学习代码仍由学生在 Notebook 中完成。"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import tempfile

import ipywidgets as w
import torch

from theme import chip, frame as card

EVENTS = {
    'observe': ('观察真实分词', 5),
    'guess_round': ('猜 token 数：打完一轮', 10),
    'row_prediction': ('预测改行影响并核对', 10),
    'counterexample': ('旧版：找到分段计数反例', 15),
    'cut_round': ('旧版：切一刀打完一轮', 10),
    'compare': ('计数函数通过检查', 20),
    'density': ('token 密度函数通过检查', 20),
    'lookup': ('选做深入：两种查表方法通过检查', 20),
    'row': ('选做深入：改行与定位通过检查', 20),
    'memory': ('选做深入：one-hot 资源估算通过检查', 20),
    'mask': ('选做深入：有效位置平均通过检查', 20),
}
CORE = ('compare', 'density')
OPTIONAL = ('lookup', 'row', 'memory', 'mask')
# 旧版关卡已合并进「猜 token 数」，只保留已有记录，不再新增。
LEGACY = ('counterexample', 'cut_round')
REQUIRED_MAX = sum(EVENTS[k][1] for k in ('observe', 'guess_round', 'row_prediction') + CORE)
class Progress:
    """按固定事件计分；读取旧记录后原子替换，不因重跑或重开内核清零。"""
    def __init__(self, path):
        self.path = Path(path)
        self.read()  # 损坏的文件必须报错，不能静默清空历史。

    def read(self):
        if not self.path.exists():
            return {'version': 1, 'events': {}, 'reflection': ''}
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if (data.get('version') != 1 or not isinstance(data.get('events'), dict)
                or not isinstance(data.get('reflection'), str)):
            raise ValueError(f'进度文件格式不兼容，请保留原文件并检查：{self.path}')
        if any(k not in EVENTS or not isinstance(v, dict) for k, v in data['events'].items()):
            raise ValueError('进度包含无法识别的事件，请保留文件并检查。')
        if not isinstance(data.get('games', {}), dict):
            raise ValueError('进度中的关卡成绩格式不兼容，请保留文件并检查。')
        return data

    def best(self, game):
        return self.read().get('games', {}).get(game)

    def record_best(self, game, score):
        """只在打破纪录时写入；返回 (是否新纪录, 当前最好成绩)。"""
        data = self.read()
        games = data.setdefault('games', {})
        old = games.get(game)
        if old is not None and score <= old['score']:
            return False, old['score']
        games[game] = {'score': score, 'at': datetime.now(timezone.utc).isoformat()}
        self._write(data)
        return True, score

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                         prefix='.progress-', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        try:
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def award(self, event, evidence, assistance='实验操作'):
        if event not in EVENTS:
            raise ValueError('未知挑战')
        data = self.read()
        fresh = event not in data['events']
        old = data['events'].get(event, {})
        record = {'at': datetime.now(timezone.utc).isoformat(), 'evidence': evidence,
                  'assistance': assistance}
        data['events'][event] = {'first': old.get('first', record), 'latest': record}
        self._write(data)
        return fresh

    def reflection(self, text):
        data = self.read()
        data['reflection'] = text
        self._write(data)


def matrix_html(table, labels, highlights=()):
    highlights = set(highlights)
    rows = []
    for i, row in enumerate(table.tolist()):
        cells = ''.join(f'<td style="background:rgba({"45,164,78" if n >= 0 else "130,80,223"},'
                        f'{min(.45, .08 + abs(n)*.12):.3f})">{n:.3f}</td>' for n in row)
        rows.append(f'<tr style="outline:{"2px solid #2da44e" if i in highlights else "none"}">'
                    f'<th>{escape(str(labels[i]))}</th>{cells}</tr>')
    return '<table><thead><tr><th>行</th>' + ''.join(f'<th>d{j}</th>' for j in range(table.shape[1])) + \
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'


class LessonLab:
    def __init__(self, tokenizer, progress_path):
        self.tokenizer = tokenizer
        self.progress = Progress(progress_path)
        self.motion = w.Checkbox(value=True, description='播放短动画（无声音）', indent=False)
        self.score = w.HTML()
        self.meter = w.HTML()
        self.feedback = w.HTML()
        self.refresh = w.Button(description='刷新进度', icon='refresh')
        self.refresh.on_click(lambda _: self._guard(self._render_progress))
        self.dashboard = w.VBox([self.score, w.HBox([self.motion, self.refresh]), self.feedback])
        self._baseline = None
        self._render_progress()

    def _render_progress(self):
        data = self.progress.read()
        events = data['events']
        points = sum(EVENTS[k][1] for k in events)
        complete = sum(k in events for k in CORE)
        self.meter.value = card(f'<b>{points} 探索积分 · 必做代码 {complete}/{len(CORE)}</b>　<small>解释与迁移待讨论</small>')
        rows = []
        for key in CORE + tuple(k for k in OPTIONAL if k in events):
            record = events.get(key)
            status = '代码检查通过' if record else '待完成代码'
            detail = (record['latest']['assistance'] + '；' + record['latest']['evidence']) if record else '尚无检查记录'
            rows.append(f'<tr><td>{escape(EVENTS[key][0].replace("通过检查", ""))}</td>'
                        f'<td>{status}</td><td>{escape(detail)}</td></tr>')
        reflection = '已保存，待讨论' if data['reflection'].strip() else '待填写与讨论'
        legacy = sum(EVENTS[k][1] for k in LEGACY if k in events)
        self.score.value = card('<h3>课程 5 · token 账算得清，embedding 说得清</h3>'
            f'<span class="metric">{points}</span> 探索积分　｜　必做代码 {complete}/{len(CORE)}'
            '<div class="chips">' + ''.join(chip(label, event in events) for event, label in
                [('observe', '分词观察'), ('guess_round', '猜数一轮'), ('density', '密度测量'), ('row_prediction', '查表预测')]) + '</div>'
            '<table><tr><th>能力证据</th><th>状态</th><th>最近检查</th></tr>' + ''.join(rows) + '</table>'
            f'<p>解释与迁移：{reflection}。程序检查不等于独立掌握；是否用过提示由你自报。</p>'
            f'<small>必做最高 {REQUIRED_MAX} 积分'
            + (f'，另有旧版关卡留下的 {legacy} 分照常计入' if legacy else '')
            + '；选做深入 Notebook 另有 80 分。积分不是考试成绩。不扣分、不因中断清零，同一挑战只加一次。记录只在本地。</small>')

    def _message(self, text, success=False):
        animation = ' pop' if success and self.motion.value else ''
        self.feedback.value = card(f'<div class="win{animation}">{escape(text)}</div>')

    def _guard(self, action):
        try:
            action()
        except Exception as exc:
            self._message(f'还没有完成这一步，我们先检查这个差异：{type(exc).__name__}：{exc}。改好后可以再试，已有积分保留。')

    def _award(self, key, evidence, assistance='实验操作'):
        fresh = self.progress.award(key, evidence, assistance)
        self._render_progress()
        self._message((f'+{EVENTS[key][1]}　' if fresh else '复验通过，不重复加分。') + evidence, True)

    def embedding(self, E, ids, vocab):
        self._baseline = E.detach().clone()
        base = self._baseline
        seq = ids.detach().clone()
        self.row_id = w.Dropdown(options=[(f'{i} · {name}', i) for i, name in enumerate(vocab)], description='选一行')
        self.position = w.IntSlider(value=0, min=0, max=len(seq)-1, description='输入位置', continuous_update=False)
        self.delta = w.FloatSlider(value=1, min=-3, max=3, step=.5, description='第 0 维增加', continuous_update=False)
        self.prediction = w.Text(placeholder='如 1,3；从 0 编号；没有变化填 无', description='预测位置', layout=w.Layout(width='95%'))
        self.row_button = w.Button(description='揭晓改行影响', button_style='info')
        self.embedding_view = w.HTML()
        self.row_result = w.HTML()

        def render(_=None):
            row = self.row_id.value
            p = self.position.value
            selected = int(seq[p])
            # 演示副本不修改 Notebook 的 E，也不替学生完成改行题。
            self.embedding_view.value = card('<h3>一张表怎样被反复使用</h3>'
                '<small>随机教学表，未训练；颜色表示数值，不能解释为语义属性。</small>'
                '<p>输入位置 → token / ID</p><div class="chips">' + ''.join(chip(f'{i}: {vocab[int(v)]} / {int(v)}', i == p) for i, v in enumerate(seq))
                + '</div><p>表 E：绿色边框是当前输入位置取出的那一行</p>'
                + matrix_html(base, [f'{i} · {t}' for i, t in enumerate(vocab)], [selected])
                + f'<p>位置 {p} 的 one-hot：' + ' '.join(str(int(i == selected)) for i in range(len(vocab)))
                + '</p><p>它与 E 相乘，只有对应行留下：</p>'
                + matrix_html(base[[selected]], [f'输出位置 {p}'])
                + f'<p>下一步：只改变 ID {row} 的第 0 维，增加 {self.delta.value:g}，先写哪些输入位置受影响。</p>')
            self.row_result.value = ''

        for control in [self.row_id, self.position, self.delta]:
            control.observe(render, names='value')

        def reveal():
            raw = self.prediction.value.strip()
            if not raw:
                raise ValueError('先填预测，再揭晓；预测错也可以重新试')
            guessed = [] if raw == '无' else [int(x.strip()) for x in raw.replace('，', ',').split(',')]
            if len(set(guessed)) != len(guessed) or any(i < 0 or i >= len(seq) for i in guessed):
                raise ValueError('位置不可重复，且必须在输入长度范围内')
            changed = base.clone()
            changed[self.row_id.value, 0] += self.delta.value
            before, after = base[seq], changed[seq]
            actual = [i for i in range(len(seq)) if not torch.equal(before[i], after[i])]
            self.row_result.value = card(f'<h3>实际变化的位置：{actual}</h3>'
                + matrix_html(after, [f'位置 {i} · {vocab[int(v)]}' for i, v in enumerate(seq)], actual)
                + '<p>这里只改演示副本，课堂的 E 未改动。回到正文，用自己的话说明为什么改一行会影响这些位置。</p>')
            if sorted(guessed) == actual and self.delta.value != 0:
                self._award('row_prediction', f'ID {self.row_id.value} 改行预测正确，影响位置 {actual}。')
            else:
                self._message('这次预测与结果不同，或增量为 0。我们选一个位置，沿着它的 ID 找到表里的行；再检查其他位置有没有使用同一行。可以修改后重试。')

        self.row_button.on_click(lambda _: self._guard(reveal))
        render()
        return w.VBox([self.position, w.HBox([self.row_id, self.delta]), self.embedding_view,
                       self.prediction, self.row_button, self.row_result, self.feedback])

    def memory(self):
        self.mem_controls = {
            'B': w.IntSlider(value=4, min=1, max=16, description='条数 B', continuous_update=False),
            'T': w.IntSlider(value=2048, min=256, max=8192, step=256, description='长度 T', continuous_update=False),
            'V': w.IntSlider(value=150000, min=10000, max=200000, step=10000, description='词表 V', continuous_update=False),
            'D': w.IntSlider(value=768, min=128, max=2048, step=128, description='维度 D', continuous_update=False),
        }
        self.limit = w.FloatSlider(value=2, min=.5, max=16, step=.5, description='额度 GB', continuous_update=False)
        self.memory_view = w.HTML()

        def render(_=None):
            B, T, V, D = (self.mem_controls[k].value for k in ('B', 'T', 'V', 'D'))
            sizes = {'ID / int64': B*T*8, '表 / float32': V*D*4,
                     '输出 / float32': B*T*D*4, '额外 one-hot / float32': B*T*V*4}
            shared = B*T*8 + V*D*4 + B*T*D*4
            bars = []
            for name, value in [('直接查表', shared), ('保留 ID 并显式创建 one-hot', shared+B*T*V*4)]:
                gb = value / 1e9
                width = min(100, gb / self.limit.value * 100)
                color = '#e5534b' if gb > self.limit.value else '#2da44e'
                bars.append(f'<p>{name}：{gb:.4f} GB / {self.limit.value:g} GB　'
                            + ('超过此额度' if gb > self.limit.value else '这些张量在此额度内')
                            + f'</p><div class="meter"><div class="fill" style="width:{width}%;background:{color}"></div></div>')
            self.memory_view.value = card('<h3>把内存账摆出来</h3>' + ''.join(bars)
                + '<table><tr><th>组成</th><th>字节</th><th>GB（10⁹）</th></tr>'
                + ''.join(f'<tr><td>{name}</td><td>{n:,}</td><td>{n/1e9:.6f}</td></tr>' for name, n in sizes.items())
                + '</table><p>保持 V、D 不变，把 T 翻倍：哪块没有变？再单独把 D 翻倍。</p>'
                '<small>只做整数估算，不分配大张量。这里假设 ID、表和输出保留，两方案共享；不含模型其余部分、梯度、优化器、缓存及临时 int64 one-hot。不是完整运行峰值，额度也不是本机可用内存实测。</small>')

        for control in [*self.mem_controls.values(), self.limit]:
            control.observe(render, names='value')
        render()
        return w.VBox([*self.mem_controls.values(), self.limit, self.memory_view])

    def checkpoint(self, key, get_submission):
        """按钮点击时重新取 Notebook 中的函数，修改后不用重建整个实验室。"""
        mode = w.Dropdown(options=['未声明提示使用情况', '自报：借助提示完成', '自报：独立完成'], description='本次方式', layout=w.Layout(width='95%'))
        button = w.Button(description='检查我的代码', button_style='success')

        def check():
            evidence = self.verify(key, get_submission())
            self._award(key, evidence, mode.value)

        button.on_click(lambda _: self._guard(check))
        return w.VBox([mode, button, self.feedback, self.meter])

    def verify(self, key, submission):
        """多输入验证。通过只能证明这些用例成立，不自动判定文字解释或独立性。"""
        if key == 'compare':
            for parts in [['hel', 'lo'], ['token', 'ization'], ['你', '好'], [], ['', 'a'], ['第一段\n', '第二段']]:
                result = submission(parts, self.tokenizer)
                text = ''.join(parts)
                joined = self.tokenizer.encode(text, add_special_tokens=False).ids
                pieces = [i for p in parts for i in self.tokenizer.encode(p, add_special_tokens=False).ids]
                expected = dict(text=text, joined_ids=joined, piece_ids=pieces,
                                joined_count=len(joined), piece_count=len(pieces))
                for field, value in expected.items():
                    if result[field] != value:
                        raise AssertionError(f'片段 {parts!r} 的 {field} 不正确，请对照两种操作顺序')
            return '通过 6 组输入，包括空片段列表和换行；原因解释待讨论。'
        if key == 'density':
            samples = {'英文': 'The vendor shall submit a report.', '中文': '乙方应提交季度报告。',
                       '换行': '第一行\n第二行', 'emoji': '🙂🙂'}
            result = submission(self.tokenizer, samples)
            if [row['name'] for row in result] != list(samples):
                raise AssertionError('结果顺序或 name 字段与输入不一致')
            for row, (name, text) in zip(result, samples.items()):
                tokens = len(self.tokenizer.encode(text, add_special_tokens=False).ids)
                if row['chars'] != len(text) or row['tokens'] != tokens:
                    raise AssertionError(f'{name} 的 chars 或 tokens 不正确；tokens 要用 add_special_tokens=False')
                if abs(row['tokens_per_char'] - tokens / len(text)) > 1e-9:
                    raise AssertionError(f'{name} 的 tokens_per_char 应为 token 数 ÷ 字符数')
            if submission(self.tokenizer, {}) != []:
                raise AssertionError('没有样本时应返回空列表')
            try:
                submission(self.tokenizer, {'空': ''})
            except ValueError:
                return '通过 4 种文本、空输入与空文本报错；结论与适用范围待讨论。'
            raise AssertionError('空文本没有字符，比例没有意义，应抛出 ValueError')
        if key == 'lookup':
            generator = torch.Generator().manual_seed(51)
            for V, D, ids in [(7, 3, [4, 1, 4, 0, 6]), (5, 6, [2, 0]), (3, 2, [])]:
                E = torch.randn(V, D, generator=generator)
                ids = torch.tensor(ids, dtype=torch.long)
                original = E.clone()
                left, right = submission(E, ids)
                for result in [left, right]:
                    torch.testing.assert_close(result, original[ids])
                torch.testing.assert_close(E, original)
            return '通过 3 组形状/重复 ID/空序列；请继续解释 one-hot 的 0 和 1。'
        if key == 'row':
            E, ids, changed, positions, target, delta = submission
            if self._baseline is None:
                raise ValueError('先运行上方矩阵实验台，记录未修改的 E')
            torch.testing.assert_close(E, self._baseline)
            expected = E.clone()
            expected[target] += delta
            torch.testing.assert_close(changed, expected)
            before, after = E[ids], expected[ids]
            actual = [i for i in range(len(ids)) if not torch.equal(before[i], after[i])]
            if list(positions) != actual or not actual:
                raise AssertionError('请检查改行与定位结果，需要实际产生变化并按位置顺序列出')
            return f'原 E 保持不变，只改指定行；变化位置 {actual}。新条件迁移仍待讨论。'
        if key == 'memory':
            for B, T, V, D in [(4, 2048, 150000, 768), (2, 17, 31, 6), (1, 5, 7, 3)]:
                actual = submission(B, T, V, D)
                expected = {'ids': B*T*8, 'one_hot': B*T*V*4, 'table': V*D*4, 'output': B*T*D*4}
                for name, value in expected.items():
                    if actual[name] != value:
                        raise AssertionError(f'{name} 不正确；条件 B={B},T={T},V={V},D={D}，返回单位应为字节')
            return '通过 3 组 B/T/V/D；单位与共享表计算正确，估算范围解释待讨论。'
        if key == 'mask':
            vectors = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
            mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
            expected = torch.stack([vectors[0, :2].mean(0), vectors[1].mean(0)])
            torch.testing.assert_close(submission(vectors, mask), expected)
            padded = torch.cat([vectors, torch.ones(2, 2, 3)*999], dim=1)
            extended = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
            torch.testing.assert_close(submission(padded, extended), expected)
            mask[0] = False
            try:
                submission(vectors, mask)
            except ValueError:
                return '选做通过：排除补齐、追加补齐不变、拒绝空有效序列。'
            raise AssertionError('一条文本没有有效位置时，应抛出 ValueError')
        raise ValueError('未知代码检查')

    def reflection_panel(self):
        self.notes = w.Textarea(value=self.progress.read()['reflection'],
            placeholder='我现在能解释什么？引用一次实验结果。还有什么条件没有验证？也可以写 Notebook 中答案的位置。',
            layout=w.Layout(width='98%', height='130px'))
        save = w.Button(description='保存我的解释', icon='save')

        def persist():
            if not self.notes.value.strip():
                raise ValueError('先写下解释或待讨论的问题')
            self.progress.reflection(self.notes.value)
            self._render_progress()
            self._message('解释已保存在本地，状态为待讨论。不会按字数或关键词自动认定掌握。', True)

        save.on_click(lambda _: self._guard(persist))
        return w.VBox([w.HTML(card('<h3>我现在能讲清什么？</h3><p>上面的面试回答留在 Notebook 里；这里保存你想带到讨论中的总结。每个挑战的最近证据也会保留。</p>')),
                       self.notes, save, self.feedback, self.score])
