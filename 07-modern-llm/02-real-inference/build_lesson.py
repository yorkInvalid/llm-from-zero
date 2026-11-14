"""重新生成课件源文件会清空已执行输出；一般学习时直接打开 lesson.ipynb。"""
import base64
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

def slide(filename, title, boxes, arrows, notes):
    fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0,15), ylim=(0,8)); ax.axis('off')
    ax.text(.35,7.5,title,fontsize=23,weight='bold',color='#18354b')
    colors = ['#e0edf8','#fff0bd','#def0e5','#f7dfd8']
    for i,(x,y,w,h,label) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=colors[i%4],edgecolor='#41586b',linewidth=1.6))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=15)
    for a,b in arrows:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','lw':2,'color':'#41586b'})
    for x,y,label in notes:
        ax.text(x,y,label,fontsize=14,color='#34495e',va='top')
    fig.savefig(HERE/'assets'/filename,bbox_inches='tight'); plt.close(fig)

slide('inference-flow.png','02 · 从一句话，到模型生成的下一枚 token',[
 (.5,5.25,3.7,1.3,'messages → chat template\n角色标记 + 用户文本'),
 (5,5.25,3.5,1.3,'Tokenizer → input_ids\n[1, T]，每格是整数 ID'),
 (9.3,5.25,5,1.3,'Embedding → 28 个 Block\n[1,T,1024] → [1,T,1024]'),
 (9.3,2.65,5,1.3,'final RMSNorm → LM head\nlogits：[1,T,151936]'),
 (5,2.65,3.5,1.3,'只取最后一个位置\n[1,151936] → 选 1 个 ID'),
 (.5,2.65,3.7,1.3,'追加这个 ID → 再预测\n直到 EOS 或长度上限'),
], [((4.3,5.9),(4.9,5.9)),((8.6,5.9),(9.2,5.9)),((11.8,5.15),(11.8,4.05)),((9.2,3.3),(8.6,3.3)),((4.9,3.3),(4.3,3.3))],
[(.5,1.6,'B=1：一次处理一条文本；T：模板展开后的 token 数，不是汉字数。'),(.5,1.0,'推理：参数 W 固定。变化的是输入、hidden、Q/K/V、logits 和缓存。'),(.5,.4,'训练课再加入 labels、loss、backward 和 optimizer.step；这一课先把推理走通。')])
slide('cache-flow.png','KV cache · 新 token 逐个进入，历史 K/V 留在每一层',[
 (.5,5.0,6,1.4,'第 1 次：prefill\n输入 x₁ … x_T → 预测 y₁'),
 (8.4,5.0,6,1.4,'28 层分别保存 K、V\n每层各 [1,8,T,128]'),
 (.5,2.8,6,1.4,'第 2 次：decode\n只输入 y₁ → 预测 y₂'),
 (8.4,2.8,6,1.4,'追加 y₁ 的 K、V\n每层各 [1,8,T+1,128]'),
], [((6.6,5.7),(8.3,5.7)),((3.5,4.9),(3.5,4.3)),((11.4,4.9),(11.4,4.3)),((6.6,3.5),(8.3,3.5))],
[(.5,2.1,'decode 的 Q：[1,16,1,128]；它仍要读取所有历史 K/V。'),(.5,1.5,'有缓存：输入长度 T, 1, 1, …；无缓存：T, T+1, T+2, …'),(.5,.9,'注意：刚预测出的 y₂ 还没进入模型，此时 cache 长度是 T+1，不是 T+2。'),(.5,.3,'缓存节省重复计算，不改变训练好的参数，也不保证短输入上的速度必然提高。')])

cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s): cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]})
def diagram(name):
    data=base64.b64encode((HERE/'assets'/name).read_bytes()).decode()
    cells.append({'cell_type':'markdown','metadata':{},'source':[f'![教学结构图](attachment:{name})'],'attachments':{name:{'image/png':data}}})

md('''# 07 · 第二课：让真实 Qwen 一步一步生成回答

上一课已经读过结构，现在把真实参数加载进来。我们使用同一版本的 **Qwen3-0.6B**，在本机处理一句中文，观察输入、张量、下一 token 和完整回答。

你打开就能看到教师执行保存的结果；这些是实际模型输出，不是预设答案，也不代表你已经完成实践。按顺序运行时，你可以直接改 `QUESTION` 看变化。无需租 GPU、无需 API key；首次需要下载约 1.2 GB 权重，保存在 Hugging Face 本机缓存，不进入 Git。

本课路线：**先看图 → 真实回答 → 拆输入 → 拆一次 forward → 手动生成 → KV cache 对照 → 速度与内存**。不设猜数字和填空题。''')
diagram('inference-flow.png')
md('''## 0 · 准备环境，只加载一次

在仓库根目录的终端运行：
```bash
uv sync --extra inference
```
然后选择本项目的 `llm-from-zero` 内核。下面打印的 Python 路径应指向项目 `.venv`。如果刚安装依赖，请重启内核再从头运行。

本课固定 Transformers 4.57.3 和模型 revision；不执行模型仓库的远程 Python。默认优先使用 Mac 的 MPS，无法使用时回到 CPU。为观察缓存路径的数值一致性使用 float32：权重约 2.22 GiB，实际进程内存还包括激活、缓存和框架开销。''')
code('''from pathlib import Path
from datetime import datetime
import sys, json, platform, gc
import torch
import transformers
from IPython.display import display, Markdown

# 从仓库根目录或课件目录启动都能找到工具。
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / '07-modern-llm').is_dir())
LESSON = ROOT / '07-modern-llm' / '02-real-inference'
sys.path.insert(0, str(LESSON))
from inference_utils import MODEL_ID, REVISION, manual_generate, check_cache, benchmark, eos_ids

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print('Python:', sys.executable)
print('torch:', torch.__version__, '| transformers:', transformers.__version__)
print('设备:', device, '| dtype: float32')''')
code('''from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=REVISION, trust_remote_code=False,
    dtype=torch.float32, attn_implementation='sdpa',
).to(device)
model.eval()  # 关闭训练行为；下面的 inference_mode 另外关闭梯度记录。
print('真实模型:', type(model).__name__)
print('参数量:', f'{sum(p.numel() for p in model.parameters()):,}')
print('层数:', len(model.model.layers), '| 共享词表权重:', model.lm_head.weight is model.model.embed_tokens.weight)''')
md('''## 1 · 先看到真实回答

这里的 Qwen3-0.6B 是经过后训练、支持对话与思考模式的模型，不能把它当作随机初始化网络或纯 Base 模型。

先关闭思考模式，便于看清普通回答的生成。`enable_thinking=False` 控制的是模板/生成行为，不是删除网络层，也不是证明模型内部“不推理”。本课不训练、不更新 W。''')
code('''QUESTION = '请用一句话解释什么是自回归生成。'
messages = [{'role': 'user', 'content': QUESTION}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
# 模板已经包含特殊 token，不能再重复添加。
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(device)
input_ids = inputs['input_ids']
T = input_ids.shape[1]
# 显式覆盖模型自带的采样设置；None 表示本次不用这些采样参数。
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                               temperature=None, top_p=None, top_k=None, use_cache=True)
answer_ids = generated[0, T:]
answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
print('问题:', QUESTION)
print('模型回答:', answer)
print('输入/新生成 token 数:', T, '/', len(answer_ids))
print('停止原因:', 'EOS' if answer_ids[-1].item() in eos_ids(model) else '达到 64 token 上限（可能未说完）')''')
md('''## 2 · 模型实际收到的不是 Python 消息字典

`messages` 是应用接口的数据结构。chat template 把角色、内容和轮次边界拼成文本，tokenizer 再转换成整数。

刚才的回答可以说明模型生成成功，但“通过历史数据生成未来数据”的说法比较笼统；对本课，更准确的是“根据已有 token 前缀逐个预测下一个 token”。我们保留原始输出，不把它修改成标准答案。

下面依次显示：原始文字、带角色标记的文本、真正进入模型的 ID。**T 包含角色标记和模板内容，不等于汉字数。** 模板末尾的空 `<think>` 段来自关闭思考模式的设置，不是本次模型生成的思维链。''')
code('''print('原始用户文字:', QUESTION)
print('模板展开后:\\n' + prompt)
print('input_ids.shape:', tuple(input_ids.shape))
print('input_ids:', input_ids[0].tolist())
print('attention_mask:', inputs['attention_mask'][0].tolist())
print('\\n逐 token 查看（tokenizer 的原始片段不一定是可读汉字）：')
for position, token_id in enumerate(input_ids[0].tolist()):
    print(f'{position:2d}  ID={token_id:6d}  piece={tokenizer.convert_ids_to_tokens(token_id)!r}')''')
md('''本例只有一条输入、没有 padding，所以 `attention_mask` 全为 1。它不等于“允许看未来”：模型内部仍然应用 causal mask。

一个汉字可能跨多个 token。单独 decode 某个 token 可能出现替换符；观察生成文本时，应 decode **整段累计 ID**。

Base 模型通常用普通文本续写；聊天模型通常按它训练时的模板输入。给同一个聊天模型直接喂裸文本，并不等于把它变回 Base。我们在此不额外下载第二个模型。''')
md('''## 3 · 拆开一次 forward：每一层到底收到了什么

`model(...)` 只算预测分数，不会自己一直说下去。下面临时注册观察函数（hook），记录首层投影的实际输出形状；观察后立即移除。

B=1，T 取上面实际长度。Q 投影宽度 2048，K/V 投影宽度 1024；Q/K Norm 前先拆头，Norm 的输出排列仍是 `[B,T,头数,128]`，Attention 内部再转成 `[B,头数,T,128]`。''')
code('''shapes = {}
def record(name):
    def hook(module, args, output):
        shapes[name] = list(output.shape)
    return hook

first = model.model.layers[0]
watched = {
    'Embedding': model.model.embed_tokens,
    '首层 RMSNorm': first.input_layernorm,
    'Q projection': first.self_attn.q_proj,
    'K projection': first.self_attn.k_proj,
    'V projection': first.self_attn.v_proj,
    'Q head norm': first.self_attn.q_norm,
    'K head norm': first.self_attn.k_norm,
    'SwiGLU gate': first.mlp.gate_proj,
    'SwiGLU down': first.mlp.down_proj,
    '最终 RMSNorm': model.model.norm,
    'LM head': model.lm_head,
}
handles = [module.register_forward_hook(record(name)) for name, module in watched.items()]
try:
    with torch.inference_mode():
        forward = model(**inputs, use_cache=False)
finally:
    for handle in handles:
        handle.remove()
for name, shape in shapes.items():
    print(f'{name:18s} {shape}')
assert forward.logits.shape == (1, T, model.config.vocab_size)''')
md('''## 4 · 为什么只看最后一个位置？

完整 logits 为 `[1,T,151936]`。第 i 个位置预测它后面的 token。用户输入已经确定；要继续写，就取最后一个位置的分数 `[1,151936]`。

logit 是实数分数，还不是概率。softmax 转成概率供查看或采样；**贪心 argmax 可以直接对 logits 取最大值**，因为 softmax 不改变排序。''')
code('''next_logits = forward.logits[:, -1, :]
probabilities = next_logits.float().softmax(dim=-1)
values, indices = probabilities[0].topk(8)
print('最后位置 logits.shape:', tuple(next_logits.shape))
print('下一 token 候选（不是整句答案的概率）：')
for probability, token_id in zip(values.tolist(), indices.tolist()):
    print(f'ID={token_id:6d}  p={probability:.4f}  piece={tokenizer.convert_ids_to_tokens(token_id)!r}')
print('贪心选择:', indices[0].item())
assert indices[0].item() == answer_ids[0].item()
del forward, next_logits, probabilities  # 不保留整段大词表 logits。''')
md('''## 5 · 不用 generate，亲手走 5 步

先不使用 cache，直接反复输入完整前缀。下面六行核心操作就是自回归生成：forward → 取最后位置 → 选 ID → 追加 → 再次 forward。

新生成的 token 作为下一步输入。训练时的标准 teacher forcing 则输入真实文本前缀，目标是后移一位的真实 token；不是把本课循环原封不动变成训练。''')
code('''ids = input_ids.clone()
with torch.inference_mode():
    for step in range(5):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        old_length = ids.shape[1]
        ids = torch.cat([ids, next_id], dim=1)
        print(f'第 {step+1} 步，送入 {old_length} 个 token，选中 ID={next_id.item()}')
        print('  累计生成:', repr(tokenizer.decode(ids[0, T:], skip_special_tokens=True)))
        if next_id.item() in eos_ids(model):
            break
assert torch.equal(ids[0, T:], answer_ids[:ids.shape[1]-T])
print('与 generate 的贪心前缀一致。')
del output''')
md('''`logits_to_keep=1` 只让输出投影保留最后一个位置的 logits，减少词表分数的内存；没有 cache 时，前面的 Transformer 仍然处理完整前缀。

EOS 也是词表中的 ID，由模型像其他 token 一样预测出来。停止逻辑在循环外层检查它；达到 `max_new_tokens` 只是人为截断，不代表模型认为回答结束。''')
md('''## 6 · 同一份参数，为什么回答可以不同？

贪心每步选最大分数。采样先把 logits 除以 temperature，再只保留 top-k 候选、归一化并随机抽取。temperature 越小通常越集中；它不改变 W，也不等于事实正确率。

下面固定随机种子，只演示一次采样；不同设备和软件版本仍可能产生数值与输出差异。采样公式在 `inference_utils.py` 的 `choose_token` 中。''')
code('''sampled = manual_generate(model, input_ids, max_new_tokens=64, use_cache=True,
                          sample=True, temperature=0.7, top_k=20, seed=42)
print('贪心:', answer)
print('采样:', tokenizer.decode(sampled['new_ids'][0], skip_special_tokens=True))
print('采样停止原因:', sampled['stop_reason'])''')
md('''## 7 · KV cache：保留历史计算

每层分别保存历史 K/V。注意：缓存不是可学习参数，也不是把整个回答存起来检索。首次完整处理 prompt 称为 prefill，后续每步处理一个新 token 称为 decode。''')
diagram('cache-flow.png')
code('''# 先观察真实缓存，而不是仅按公式猜尺寸。
with torch.inference_mode():
    prefill = model(**inputs, use_cache=True, logits_to_keep=1)
cache = prefill.past_key_values
k0, v0 = cache[0]
print('缓存类型:', type(cache).__name__, '| 层数:', len(cache))
print('第 0 层 K:', tuple(k0.shape), '| V:', tuple(v0.shape))
print('prefill 后 cache 长度:', cache.get_seq_length())
assert tuple(k0.shape) == (1, 8, T, 128)
cache_bytes = sum(k.numel()*k.element_size() + v.numel()*v.element_size() for k, v in cache)
print(f'全部 28 层 KV 数据: {cache_bytes / 2**20:.3f} MiB')
del prefill, cache, k0, v0

cached_run = manual_generate(model, input_ids, max_new_tokens=5, use_cache=True, verbose=True, tokenizer=tokenizer)''')
md('''上一格的第二步开始，`input_tokens` 应为 1。工具内部新增的关键行如下：

```python
current = ids[:, -1:] if cache is not None else ids
output = model(input_ids=current,
               attention_mask=torch.ones_like(ids),  # 长度包括历史 + 当前
               past_key_values=cache, use_cache=True,
               logits_to_keep=1)
cache = output.past_key_values
```

模型根据 cache 长度衔接新 token 的位置，RoPE 不能每一步都从位置 0 开始。本例无 padding；若以后支持批处理、左 padding 或复杂缓存，需要明确处理位置和 mask，不能简单照抄。

为什么能复用？因果注意力下，历史位置不能看到后来 token，`eval()` 下它们的表示不会因追加未来 token 改变。每层当前 Q 仍会读取这一层的历史 K/V。''')
md('''## 8 · 验证优化前后算的是同一件事

不能只看两句话“意思差不多”。我们让两条路径输入完全相同的 token 前缀，连续 3 步比较整个词表的 logits，同时检查缓存长度。浮点运算顺序不同允许小误差；这不是训练质量评估。''')
code('''cache_errors = check_cache(model, input_ids, steps=3)
print('每一步最大的 logit 绝对误差:', cache_errors)
print('通过：完整前缀与缓存路径在本课 float32 容差内一致，缓存长度正确。')''')
md('''## 9 · 实测速度：不提前宣布“快了多少倍”

固定生成 12 步、两种路径分别预热、各测 3 次并取中位数。计时前后同步 MPS/CUDA，排除异步提交造成的虚假速度；包括 prefill 与 decode，不包含下载、分词和打印。

为了固定工作量，这个小基准忽略 EOS，不能把其文本当作正常回答。短输入下缓存管理可能抵消收益；本机结果不代表云 GPU 或长上下文性能。''')
code('''timings = benchmark(model, input_ids, tokens=12, repeats=3)
for row in timings:
    print(f"use_cache={row['use_cache']!s:5s}  中位耗时={row['median_seconds']:.3f}s  "
          f"生成吞吐={row['tokens_per_second']:.2f} token/s")
ratio = timings[0]['median_seconds'] / timings[1]['median_seconds']
print(f'本次耗时比（无缓存/有缓存）: {ratio:.2f}')''')
md('''## 10 · 内存看三笔账

1. **参数数据**：不随 prompt 长度改变；float32 权重约 2.22 GiB。
2. **KV cache 数据**：随 batch、token 数线性增加；本模型每个历史 token 的全部层 K/V 在 float32 下共 224 KiB。
3. **实际分配**：还有激活、临时缓冲与分配器等。MPS 是统一内存，下面的快照不是峰值，也不是整个 Python 进程的 RSS。

缓存公式：`2 × B × T × 28 × 8 × 128 × 每元素字节数`。GQA 保存的是 8 个 KV 头，不是展开后的 16 个 Q 头。''')
code('''weight_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
print(f'参数张量数据: {weight_bytes/2**30:.3f} GiB（共享参数只计一次）')
print(f'当前 prompt 的 KV 数据: {cache_bytes/2**20:.3f} MiB')
memory = {'parameter_bytes': weight_bytes, 'prompt_kv_bytes': cache_bytes}
if device.type == 'mps':
    torch.mps.synchronize()
    memory.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                  mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    print('MPS 分配快照:', {k: round(v/2**20, 2) for k,v in memory.items() if k.startswith('mps')}, 'MiB')

# 小型验证记录，不保存权重、cache 或整段 logits。
report = {'kind': 'teacher_execution_not_student_completion',
          'executed_at': datetime.now().astimezone().isoformat(), 'model': MODEL_ID,
          'revision': REVISION, 'torch': torch.__version__, 'transformers': transformers.__version__,
          'device': str(device), 'machine': platform.machine(), 'dtype': str(model.dtype),
          'question': QUESTION, 'prompt_tokens': T, 'answer': answer,
          'answer_tokens': len(answer_ids), 'observed_shapes': shapes,
          'cache_max_abs_errors': cache_errors, 'timings': timings, 'memory': memory}
(LESSON/'assets'/'execution-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\\n')
print('运行记录已保存：assets/execution-report.json')''')
md('''## 从这一步接到训练

现在我们能指出输入是什么、输出是什么，以及 token 是怎么一个个产生的。下一课再复用这个观察方式看训练：

| 阶段 | 输入 / 目标 | 更新什么 |
|---|---|---|
| 本课推理 | 给定 prompt，生成新 token | 不更新参数；cache 随生成增长 |
| 预训练 / 继续预训练 | 连续文本，预测真实的下一个 token | 用交叉熵和反向传播更新指定参数 |
| SFT | 对话模板，通常只对回答部分计算 loss | 更新全部参数或 LoRA 等指定参数 |
| RL | 模型生成候选，按奖励反馈优化策略 | 按相应目标更新策略参数 |

这里先停止在推理，不提前租 GPU。短输出、单次成功或较高吞吐都不构成语言能力评测；训练课会固定验证数据与指标。

需要释放模型时可以运行下面一格；之后要继续实验，应重新运行模型加载格。

参考：[固定模型版本](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) · [模型卡与非思考模式](https://huggingface.co/Qwen/Qwen3-0.6B) · [Transformers 4.57.3 Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py)。本课 helper 只支持单条无 padding 输入。''')
code('''# 可选：结束本课后取消注释运行。
# del model
# gc.collect()
# if device.type == 'mps':
#     torch.mps.empty_cache()''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'llm-from-zero','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.12'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells): c['id']=f'inference-{i:02d}'
(HERE/'lesson.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print('生成',len(cells),'个单元格')
