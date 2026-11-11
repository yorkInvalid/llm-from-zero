"""真实文件加载观察、张量观察。显示辅助，不代写课堂函数。"""
from html import escape
import ipywidgets as w
import torch

from theme import chip, frame
from tokenizer_tools import load_tokenizer

STAGES = {
    'locate': ('找到文件', '先确认我们读的是哪一个文件，位置和大小都来自本机。'),
    'read': ('读取字节', '磁盘上的文件被读成字节；此时还不能对一句话分词。'),
    'verify': ('核对内容', '把实际 SHA-256 与课程固定版本比较，确认读到的是预期文件。'),
    'parse': ('建立分词器', '把文件里的词表和规则读出来，建立能编码、解码的对象。'),
    'probe': ('试一次往返', '编码一句话，再完整还原。检查这条路径确实可以工作。'),
}


class TokenizerLoadPanel:
    def __init__(self):
        self.events = []
        self.latest = {}
        self.summary = w.HTML()
        self.details = [w.HTML() for _ in STAGES]
        self.steps = w.Accordion(children=self.details, selected_index=None)
        for i, (_, (title, why)) in enumerate(STAGES.items()):
            self.steps.set_title(i, title)
            self.details[i].value = frame(f'<p class="muted">{escape(why)}</p><p>尚未执行。</p>')
        self.widget = w.VBox([self.summary, self.steps])
        self.widget.add_class("l5-loader")
        self._render()

    def _render(self):
        final = self.latest.get('probe', {})
        failed = next((STAGES[k][0] for k, e in self.latest.items() if e.get('status') == 'error'), None)
        if final.get('status') == 'done':
            body = (f'<p><span class="hit">✓</span> 分词器就绪：<b>{final["vocab_size"]:,}</b> 个已登记 token，实际试编码通过。'
                    '<span class="muted">只加载了切分规则，没有模型权重。加载证据在下面五步里。</span></p>')
        elif failed:
            body = f'<p class="miss">✕ 停在「{escape(failed)}」这一步。展开它看原因，修好后重新运行这一格。</p>'
        else:
            body = '<p class="muted">正在加载本地分词器……</p>'
        self.summary.value = frame(body)

    def on_event(self, event):
        self.events.append(dict(event))
        key = event['stage']
        # running 之后保留路径等先前已观测信息。
        self.latest[key] = {**self.latest.get(key, {}), **event}
        current = self.latest[key]
        title, why = STAGES[key]
        text = f'<p class="muted">{escape(why)}</p>'
        if current['status'] == 'error':
            text += '<p class="miss">' + escape(current['error']) + '</p>'
        elif current['status'] == 'running':
            text += '<p>正在执行这一步。</p>'
        else:
            if key == 'locate':
                text += '<p>实际文件：<code>' + escape(current['path']) + '</code></p>'
                text += f'<p>磁盘大小：{current["size_bytes"]:,} bytes（{current["size_bytes"]/1e6:.2f} MB）</p>'
            elif key == 'read':
                text += f'<p>本次实际读取：{current["size_bytes"]:,} bytes。</p>'
            elif key == 'verify':
                text += '<p>大小与固定版本指纹均相符。</p><code>' + escape(current['sha256']) + '</code>'
            elif key == 'parse':
                text += '<table>' + ''.join(f'<tr><td>{escape(label)}</td><td>{escape(str(value))}</td></tr>' for label, value in [
                    ('model.type', current['model_type']), ('基础词表条目', current['base_vocab']),
                    ('合并规则', current['merge_rules']), ('新增 token 记录', current['added_tokens']),
                    ('预切分器类型', current['pre_tokenizer']), ('解码器类型', current['decoder'])]) + '</table>'
                text += '<p class="muted">这些数字直接来自刚读取的 JSON。新增 token 记录不一定都是基础词表之外的新 ID，不能盲目相加；总数看对象的实际登记结果。</p>'
                text += '<details><summary>文件还包含哪些一级字段？</summary><p>' + escape(', '.join(current['keys'])) + '</p></details>'
            elif key == 'probe':
                text += '<p>实际输入：' + escape(current['sample']) + '</p>'
                text += '<p>编号：' + escape(', '.join(map(str, current['ids']))) + '</p><p>完整还原：' + escape(current['restored']) + '</p>'
                text += '<p class="muted">这是一次基本检查，不表示已经验证了所有文本；也不代表模型能生成回答。</p>'
        i = list(STAGES).index(key)
        self.details[i].value = frame(text)
        self.steps.set_title(i, ('✓ ' if current['status'] == 'done' else '✕ ' if current['status'] == 'error' else '◉ ') + title)
        self._render()
        if current['status'] == 'error':
            self.steps.selected_index = i

    def load(self):
        self.events.clear()
        self.latest.clear()
        for i, (title, why) in enumerate(STAGES.values()):
            self.steps.set_title(i, title)
            self.details[i].value = frame(f'<p class="muted">{escape(why)}</p><p>尚未执行。</p>')
        self._render()
        return load_tokenizer(on_event=self.on_event)


class TensorExplorer:
    """读取实际张量，标注轴；只显示有限窗口，不修改输入、不复制大张量。"""
    def __init__(self, tensors):
        # name -> (tensor, 每个轴的含义)。由课堂提供实际对象。
        if not tensors:
            raise ValueError('至少提供一个张量')
        for name, (tensor, axes) in tensors.items():
            if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (1, 2, 3):
                raise ValueError(f'{name}：观察器接受一至三维张量')
            if len(axes) != tensor.ndim or any(n == 0 for n in tensor.shape):
                raise ValueError(f'{name}：每个轴需要含义，且长度不能为零')
            if tensor.is_complex() or tensor.layout != torch.strided:
                raise ValueError('本课观察器使用普通实数/布尔张量')
        self.tensors = tensors
        self.source = w.Dropdown(options=list(tensors), description='观察对象', layout=w.Layout(width='95%'))
        self.batch = w.IntSlider(min=0, max=0, description='第几条 B', continuous_update=False)
        self.mode = w.ToggleButtons(options=[('看整体', 'whole'), ('用整数取行', 'index'), ('用切片取行', 'slice')])
        self.row = w.IntSlider(min=0, max=0, description='选择位置/行', continuous_update=False)
        self.view = w.HTML()
        self._updating = False
        self.source.observe(self._select, names='value')
        for control in (self.batch, self.mode, self.row):
            control.observe(self._render, names='value')
        self.widget = w.VBox([self.source, self.batch, self.mode, self.row, self.view])
        self.widget.add_class("l5-tensor")
        self._select()

    def _select(self, _=None):
        self._updating = True
        tensor, _ = self.tensors[self.source.value]
        self.batch.value = 0
        self.batch.max = tensor.shape[0]-1 if tensor.ndim == 3 else 0
        self.batch.layout.display = '' if tensor.ndim == 3 else 'none'
        self.row.value = 0
        self.row.max = tensor.shape[-2] - 1 if tensor.ndim > 1 else tensor.shape[0]-1
        self.mode.value = 'whole'
        self._updating = False
        self._render()

    def selection(self):
        name = self.source.value
        tensor, axes = self.tensors[name]
        result = tensor.detach()
        axes = list(axes)
        expression = name
        if tensor.ndim == 3:
            result = result[self.batch.value]
            axes = axes[1:]
            expression += f'[{self.batch.value}]'
        if self.mode.value == 'index':
            result = result[self.row.value]
            axes = axes[1:]
            expression += f'[{self.row.value}]'
        elif self.mode.value == 'slice':
            result = result[self.row.value:self.row.value+1]
            expression += f'[{self.row.value}:{self.row.value+1}]'
        return result, axes, expression

    def _render(self, _=None):
        if self._updating:
            return
        self.row.layout.display = 'none' if self.mode.value == 'whole' else ''
        tensor, original_axes = self.tensors[self.source.value]
        result, axes, expression = self.selection()
        shape = tuple(result.shape)
        axis_tags = ''.join(chip(f'{axis} = {size}') for axis, size in zip(original_axes, tensor.shape))
        body = '<h3>同一份数据，为什么 shape 会变？</h3>'
        body += '<div class="chips">' + axis_tags + '</div>'
        body += f'<p>源张量：<b>{escape(self.source.value)}</b>　shape {tuple(tensor.shape)}　{escape(str(tensor.dtype))}　{escape(str(tensor.device))}</p>'
        body += f'<p>源张量逻辑数据量：{tensor.numel():,} × {tensor.element_size()} = {tensor.numel()*tensor.element_size():,} bytes</p>'
        body += '<p class="win"><code>' + escape(expression) + '</code> → <b class="hit">shape ' + str(shape) + '</b></p>'
        if result.ndim == 0:
            body += '<div class="chips">' + chip(result.item()) + '</div><p>现在是单个数，shape 为 ()，没有轴。</p>'
        elif result.ndim == 1:
            body += '<p>横向：' + escape(axes[0]) + '</p><div class="chips">'
            body += ''.join(chip(f'{i}: {self._number(v)}') for i, v in enumerate(result[:24].cpu().tolist())) + '</div>'
        else:
            body += '<p><b>行：' + escape(axes[0]) + '</b>　<b>列：' + escape(axes[1]) + '</b></p>'
            shown = result[:16, :12].cpu().tolist()
            body += '<div style="overflow-x:auto"><table><tr><th>行 / 列</th>' + ''.join(f'<th>{j}</th>' for j in range(min(12, result.shape[1]))) + '</tr>'
            for i, row in enumerate(shown):
                body += f'<tr><th>{i}</th>'
                for n in row:
                    color = 'rgba(45,164,78,.18)' if n > 0 else 'rgba(130,80,223,.18)' if n < 0 else 'transparent'
                    body += f'<td style="background:{color}">{escape(self._number(n))}</td>'
                body += '</tr>'
            body += '</table></div>'
        body += '<small>绿色为正、紫色为负、无底色为零；数字来自实际张量。最多显示 16 行 × 12 列或 24 项，shape 和字节数仍描述完整对象。逻辑数据量不是进程内存，也不表示索引一定分配了同等新内存。</small>'
        if self.mode.value == 'whole':
            hint = '先辨认轴的含义，再试“用整数取行”和“用切片取行”。取到的数可以一样，但轴会留下几个？把你的预测填到下方原题里。'
        elif self.mode.value == 'index':
            hint = '整数索引把这一轴的位置固定了，所以结果不再保留这一轴。切换到切片，再比较数字和 shape。'
        else:
            hint = '切片保留的是一段范围，即使范围里只有一行，这个轴仍然在。现在试着用自己的话解释它和整数索引的区别。'
        if tensor.ndim == 3:
            hint += ' 上方 B 选择器先固定一条数据，再观察它内部的位置与向量维度。'
        self.view.value = frame(body + '<p class="muted">' + escape(hint) + '</p>')

    @staticmethod
    def _number(value):
        return f'{value:.3f}' if isinstance(value, float) else str(value)
