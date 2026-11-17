# 用一道短答题逐行读 train(args)

对应云端原件 [experiment.py](../cloud/experiment.py)，第 154–288 行，每个物理行一项。可以在 IDE 分屏打开，搜索 **L253** 这样的行号。表中“→”包括返回值、变量赋值、写文件和原地状态变化；条件、循环和跨行调用不一定产生独立返回值。

**读这道题的最短路径：L164 编码 → L242 取样本 → L251 取微批 → L253 前向 → L254 算误差 → L255 累积梯度 → L263 更新 LoRA。**

本例是双问组《百家姓》中没有下面哪个姓（A.乌 / B.巫 / C.肖 / D.萧）的短答记录。原题来自传统文化题库，训练排程沿用被替换的槽位 ID，所以 ID 开头是 coig_pc。指令是“只答所选项的字母。”

    # L164 返回后的 data[8696]，ids 省略中间的提示 ID
    {'id': 'coig_pc/coig_pc_core_sample-L00593', 'source': '传统文化', 'ids': [151644, 872, 198, ..., 77091, 198, 34, 151643], 'split': 41, 'targets': 2}

| 第 3 步的尺度 | 数值 |
|---|---|
| 单题 | 41 个提示 + 2 个监督目标（C、EOS）= 43 个 token。位置从 0 算：hidden[40] 预测 ID 34；hidden[41] 预测 ID 151643。 |
| 第二个微批 | 两题长 43、53，右补齐为 [2,53]；监督位置 2+31=33。前一个微批有 435 个监督位置。 |
| 整个优化步 | 4 条，468 个监督位置；平均 loss=2.57391054。云端没有记录本题单独的 loss、概率或每个参数本步的改变量。 |

下列第 3 步数值取自 [云端测量包](../evidence/pair-measurements.tar.gz) 的 run/paired/updates.jsonl；微批形状由冻结样本长度及 active_logits 的右补齐代码推得。


## 入口、加载与数据校验

| 行 | 云端原代码 | 输入 → 输出、作用或状态变化 |
|---:|---|---|
| **L154** | `def train(args):` | 命令行解析得到的 Namespace（本例 arm=paired，另有 base、base_manifest、data、schedule、out 等路径） → 调用训练入口；正常结束隐式返回 None。产物是文件和更新后的适配器。 |
| **L155** | `out = args.out` | args.out: Path → out: Path；原实例本组为 run/paired。 |
| **L156** | `out.mkdir(exist_ok=True, parents=True)` | out → 创建输出目录及缺失的父目录；已存在也可继续。磁盘变化，无数据返回。 |
| **L157** | `data_files = [args.data/'train.jsonl', args.data/'validation.jsonl']` | args.data → data_files: list[Path]，按训练、验证的顺序指向两个 JSONL 文件。 |
| **L158** | `data_hashes = {p.name:sha(p) for p in data_files}` | 两个文件的实际字节 → data_hashes: dict[str,str]，键是文件名，值是 SHA-256。 |
| **L159** | `frozen_base = verify_base(args.base, args.base_manifest)` | 本地模型目录 + 冻结模型清单 → frozen_base: dict。检查是指定纯底座、文件符合清单；失败则中断。这时还没做前向。 |
| **L160** | `save(out/'base-verified.json', frozen_base)` | 路径 + 核验字典 → 写 base-verified.json，保留底座身份记录。 |
| **L161** | `torch.manual_seed(SEED)` | SEED=20261116 → 设置 PyTorch 随机种子；返回的 Generator 未接收。 |
| **L162** | `torch.cuda.manual_seed_all(SEED)` | 同一个种子 → 设置所有 CUDA 设备的随机种子。不能据此保证跨硬件逐位相同。 |
| **L163** | `base, tok = model_and_tokenizer(args.base)` | args.base: Path → (base, tok)：GPU 上 FP16 的 Qwen2.5-7B 模型和配套分词器。 |
| **L164** | `data, val = (encode(rows(p), tok) for p in data_files)` | 两个文件依次 rows→encode → data、val，各是 list[dict]。这里才处理文本：本题 data[8696] 得到 43 个 ID，split=41、targets=2。 |
| **L165** | `assert len(data) == 9000 and len(val) == 120` | 两个列表的长度 → 断言等于 9000、120；通过则继续，失败抛 AssertionError。 |
| **L166** | `sched=json.loads(args.schedule.read_text())` | schedule 路径 → 读文本、解析为 sched: dict，含每季度的样本顺序、监督预算和数据哈希。 |
| **L167** | `assert data_hashes['validation.jsonl']==sched['validation_sha256']` | 实际验证集哈希 + 日程中预期哈希 → 断言一致，保证两组共用同一验证集。 |
| **L168** | `if args.arm=='control': assert data_hashes['train.jsonl']==sched['control_sha256']` | arm + 训练集哈希 → 仅 control 核对原训练文件哈希；本题在 paired，本行不做该比较。 |
| **L169** | `assert {r['id']:r['targets'] for r in data}==sched['control_targets'], 'Unmatched supervised length at sample level'` | 全组的 {样本槽位ID:监督token数} + 对照表 → 断言完全相等。本题占用的槽位对应 2；比较的是长度，不是两组答案文本相同。 |
| **L170** | `assert not (set(r['id'] for r in data) & set(r['id'] for r in val))` | 训练、验证的 ID 集合 → 断言交集为空；只验证 ID，不在此处验证语义重复。 |
| **L171** | `assert sum(r['targets'] for r in data)==1353439` | 所有训练行的整数 targets → 求和并断言为 1,353,439，即每轮监督 token 数。 |
| **L172** | `assert len({r['id'] for r in data + val}) == len(data)+len(val)` | 两份列表的所有 ID → 断言不同 ID 的个数等于两份样本数之和；同时查各集合内部重复。 |
| **L173** | `assert sum(p.numel() for p in base.parameters()) == 7615616512` | base.parameters() → 累加各张量元素数，断言为 7,615,616,512；此时尚未挂 LoRA。 |

## 初始化或恢复 LoRA

| 行 | 云端原代码 | 输入 → 输出、作用或状态变化 |
|---:|---|---|
| **L174** | `last = max((step for step, name in STAGES.items() if (out/name/'state.json').exists()), default=0)` | 预定义的 8 个检查点和磁盘上的 state.json → last: int，取已保存的最大步号；新训练是 0。 |
| **L175** | `if last:` | last → 决定是否走 176–179 的恢复分支；从头训练跳过。 |
| **L176** | `ck = out/STAGES[last]` | 最新检查点步号 → ck: Path，例如步 563 对应 epoch-0.25。 |
| **L177** | `state = json.loads((ck/'state.json').read_text())` | ck/state.json → state: dict，含训练进度、数据指纹、适配器哈希等。 |
| **L178** | `assert state['data_hashes'] == data_hashes and state['adapter_sha256'] == sha(ck/'adapter_model.safetensors')` | state + 当前数据指纹和适配器文件 → 断言数据及适配器未换；不匹配不能恢复。 |
| **L179** | `model = PeftModel.from_pretrained(base, ck, is_trainable=True)` | base + ck → model: PeftModel，加载旧 LoRA 并设为可训练；这里没恢复优化器。 |
| **L180** | `else:` | last 为 0 → 进入新建 LoRA 的分支，本次从头训练经过这里。 |
| **L181** | `lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0., target_modules=['q_proj', 'k_proj', 'v_proj',` | 开始构造 LoraConfig：秩 r=8、alpha=16、dropout=0；目标列表先列注意力 q/k/v 投影。调用在下一行结束。 |
| **L182** | `'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM')` | 接上行，补 o_proj 和 MLP 的 gate/up/down 投影，任务为因果语言建模 → lora: LoraConfig（共 7 类投影）。 |
| **L183** | `model = get_peft_model(base, lora)` | base + lora → model: 带适配器的模型。原底座冻结；指定投影上增加可学习的低秩分支。 |
| **L184** | `trainable = [p for p in model.parameters() if p.requires_grad]` | model.parameters() → trainable: list[Parameter]，只保留 requires_grad=True 的参数张量。 |
| **L185** | `for p in trainable:` | trainable 列表 → 每次取一个参数对象 p，执行下一行。 |
| **L186** | `p.data = p.data.float()` | p 的底层参数值 → 转为 float32 并写回。只转换待训练参数；整个底座没有转 FP32。 |
| **L187** | `n_trainable = sum(p.numel() for p in trainable)` | trainable → n_trainable: int，按元素计数，不是张量个数。 |
| **L188** | `assert n_trainable == 20185088 and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)` | 计数 + 可训练参数名 → 断言共 20,185,088 个元素，且名称全部含 lora_。 |
| **L189** | `opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=.01)` | trainable + 初始学习率 1e−4、权重衰减 0.01 → opt: AdamW；仅这组参数交给优化器。 |
| **L190** | `scaler = torch.amp.GradScaler('cuda', init_scale=128.)` | CUDA + 初始倍率 128 → scaler: GradScaler；混合精度反传时管理缩放与溢出。 |
| **L191** | `elapsed = 0.` | 初始化 elapsed: float = 0.0 秒；累计优化步骤用时。 |
| **L192** | `log = out/'updates.jsonl'` | out → log: Path，指向 updates.jsonl；此处还没有打开文件。 |

## 恢复日志与优化器（仅续训）

| 行 | 云端原代码 | 输入 → 输出、作用或状态变化 |
|---:|---|---|
| **L193** | `if last:` | last → 如需续训，执行 194–203；新训练跳过。 |
| **L194** | `state_dict = torch.load(ck/'training_state.pt', map_location='cuda', weights_only=True)` | ck/training_state.pt → state_dict: dict，内有优化器与缩放器状态，张量映射到 CUDA。 |
| **L195** | `opt.load_state_dict(state_dict['optimizer'])` | state_dict['optimizer'] → 恢复 opt 的内部状态，包括 AdamW 的历史矩估计。 |
| **L196** | `scaler.load_state_dict(state_dict['scaler'])` | state_dict['scaler'] → 恢复 scaler 的缩放状态，以便继续混合精度训练。 |
| **L197** | `elapsed = state['train_seconds']` | state 中以秒计的 train_seconds → elapsed: float，继续累计而不是重新计时。 |
| **L198** | `del state_dict` | 临时 state_dict 变量 → 删除这个引用；已加载到 opt/scaler 的状态保留。 |
| **L199** | `valid_logs = [r for r in rows(log) if r['step'] <= last]` | 旧日志逐行数据 + last → valid_logs: list[dict]，只保留步号不超过已保存检查点的行。 |
| **L200** | `assert len(valid_logs) == last` | valid_logs 行数 + last → 断言数量相等；这里检查数量，不逐行验证顺序。 |
| **L201** | `temp_log = log.with_name('updates.valid.tmp')` | log 路径 → 临时文件路径 temp_log: Path，文件名 updates.valid.tmp。 |
| **L202** | `temp_log.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in valid_logs))` | valid_logs → 每行序列化一个 JSON 对象，写临时 JSONL，中文保留原字符。 |
| **L203** | `temp_log.replace(log)` | 临时日志路径 + 正式日志路径 → 用整理后的文件替换正式日志。 |
| **L204** | `elif log.exists():` | 没有检查点但旧日志已存在 → 进入 205；全新目录跳过。 |
| **L205** | `log.rename(out/f'updates-before-first-checkpoint-{int(time.time())}.jsonl')` | 旧 log + 当前时间戳 → 重命名留档，防止新训练和旧日志混写。 |

## 写运行配置

| 行 | 云端原代码 | 输入 → 输出、作用或状态变化 |
|---:|---|---|
| **L206** | `report = dict(base=CID, revision=REV, arm=args.arm, schedule_sha256=sha(args.schedule), quarter_samples=2250, quarter_updates=563, quarter_end_batch=2, data_hashes=data_hashes, seed=SEED, train_examples=len(data),` | 开始构造 report 字典：记录底座、版本、组别、日程指纹、每季度 2250 条/563 步、末批 2 条、数据指纹、种子和训练条数。 |
| **L207** | `validation_examples=len(val), input_plus_answer_tokens_per_epoch=sum(len(r['ids']) for r in data),` | 继续填 report：val 长度为 120；累加每行 ids 长度得到每轮非 padding 输入及答案共 1,987,444 个 token。 |
| **L208** | `answer_tokens_per_epoch=sum(r['targets'] for r in data), base_parameters=7615616512,` | 所有 targets 之和、底座参数数 → 记录监督预算 1,353,439 和底座参数数。 |
| **L209** | `trainable_parameters=n_trainable, dtype='FP16 base, FP32 adapters, AMP FP16', microbatch=MICRO,` | 记录可训练元素数、精度说明和 microbatch=2；写描述，不再改变参数精度。 |
| **L210** | `effective_batch=BATCH, epochs=2, max_len=MAX_LEN, learning_rate=LR, warmup_steps=WARMUP,` | BATCH、MAX_LEN、LR、WARMUP 等 → 记录有效批量 4、两轮、最长 384、峰值学习率 1e−4、预热 45 步。 |
| **L211** | `optimizer='AdamW', weight_decay=.01, gradient_clip=1., schedule='warmup then constant',` | 记录 AdamW、衰减 0.01、裁剪阈值 1、预热后恒定的日程；实际执行在训练循环。 |
| **L212** | `device=torch.cuda.get_device_name(), torch=str(torch.__version__), transformers=transformers.__version__,` | 当前 GPU 和库 → 记录设备名、PyTorch 与 Transformers 版本。 |
| **L213** | `peft=peft.__version__, lora_rank=8, lora_alpha=16, loss='answer-only causal cross entropy including endoftext')` | 记录 PEFT 版本、秩 8、alpha 16 与损失口径；到这里 report: dict 构造完成。 |
| **L214** | `save(out/'run-config.json', report)` | report + out → 写 run-config.json。 |

## 启动自检与基线验证（不训练）

| 行 | 云端原代码 | 输入 → 输出、作用或状态变化 |
|---:|---|---|
| **L215** | `if not last:` | last=0 → 仅新训练做 216–231 的损失自检与起始验证；恢复已有检查点时不再做。 |
| **L216** | `with torch.inference_mode():` | 进入 inference_mode 上下文 → 自检期间不建立反向传播图，不更新参数。 |
| **L217** | `model.eval()` | model → 切换评估模式；eval 本身不等同于关闭梯度。 |
| **L218** | `sample = data[0]` | data → sample=data[0]。这里实际是电影简介，不是第 8696 号索引的《百家姓》；这一步只是拿第一条做自检。 |
| **L219** | `whole = torch.tensor([sample['ids']], device='cuda')` | sample['ids']: list[int] → whole: CUDA int64 张量 [1,305]，新维度 1 表示一条样本。 |
| **L220** | `labels = whole.clone()` | whole → labels: 独立拷贝，仍为 int64[1,305]，接下来只改标签。 |
| **L221** | `labels[:, :sample['split']] = -100` | sample['split']=27 → 把 labels 开头 27 个位置设为 −100，让标准交叉熵忽略提示。whole 不变。 |
| **L222** | `ref = model(input_ids=whole, labels=labels, use_cache=False).loss.item()` | whole + labels → 模型内部执行移位和交叉熵，输出 ref: Python float；实测 2.726048946。不是《百家姓》的 loss。 |
| **L223** | `with torch.autocast('cuda', dtype=torch.float16):` | 进入 CUDA autocast 上下文 → 支持的算子按 FP16 混合精度策略执行。 |
| **L224** | `logits, targets = active_logits(model, [sample], tok.pad_token_id)` | model + 单条自检样本 + pad ID → logits: float32[278,152064]、targets: int64[278]，只取监督位置。 |
| **L225** | `part = F.cross_entropy(logits, targets).item()` | 自检 logits + targets → 对 278 个目标取平均交叉熵，得到 part: float，实测 2.726047993。 |
| **L226** | `assert abs(ref-part) < .015, (ref, part)` | ref 与 part → 断言绝对差 < 0.015；验证省显存的写法和标准模型 loss 相符。 |
| **L227** | `save(out/'reference-loss-check.json', dict(id=sample['id'], full=ref, answer_only=part))` | Sample ID + 两个 float → 写 reference-loss-check.json。 |
| **L228** | `del logits, whole, labels, targets` | 四个临时张量名 → 删除引用，不产生新张量。 |
| **L229** | `baseline = validate(model, val, tok.pad_token_id)` | model + 120 条 val + pad ID → baseline: dict，含逐样本及聚合验证统计，不更新权重。 |
| **L230** | `save(out/'validation-base.json', baseline)` | baseline + out → 写 validation-base.json，作为训练前参照。 |
| **L231** | `print('BASE VALIDATION', baseline['loss'], flush=True)` | baseline['loss'] → 立即打印起始验证 loss；不是第 3 步的训练 loss。 |

## 按日程取批，开始第 3 步

| 行 | 云端原代码 | 输入 → 输出、作用或状态变化 |
|---:|---|---|
| **L232** | `torch.cuda.reset_peak_memory_stats()` | 当前 CUDA 设备 → 重置峰值显存计数器；不会清空现有模型。 |
| **L233** | `step = 0` | 初始化 step: int = 0；恢复时也从 0 枚举，后续跳过已完成的索引。 |
| **L234** | `model.train()` | model → 切换到训练模式。参数现在还没更新。 |
| **L235** | `with log.open('a', buffering=1) as log_file:` | log: Path → 以追加模式打开 log_file；按行缓冲，退出 with 自动关闭。 |
| **L236** | `for epoch in (1, 2):` | 元组 (1,2) → epoch: int；本例第一次遇到该样本时 epoch=1。 |
| **L237** | `for indices in batches(data, epoch):` | data + epoch → batches 按冻结日程给出每批的索引列表。本例第 3 步是 [1237,3774,8696,8051]。辅助函数内部还读取命令行的全局 args.schedule。 |
| **L238** | `step += 1` | step → 加 1；本例由 2 变为 3。 |
| **L239** | `if step <= last:` | step、last → 判断是否为恢复前已完成的步；新训练第 3 步为假。 |
| **L240** | `continue` | 若 239 成立 → continue 跳到下一批；不重复更新。 |
| **L241** | `start = time.perf_counter()` | 性能计时器 → start: float，记录本步开始时刻。 |
| **L242** | `records = [data[j] for j in indices]` | indices + data → records: list[dict]，此步共 4 条。《百家姓》是 records[2]。 |
| **L243** | `total = sum(r['targets'] for r in records)` | 四条的 targets 分别为 206、229、2、31 → total: int = 468；按监督 token 数归一化。 |
| **L244** | `lr = LR*min(step/WARMUP, 1.)` | step=3 → lr=1e−4×min(3/45,1)=6.66667e−6，是此步实际学习率。 |
| **L245** | `for group in opt.param_groups:` | opt.param_groups → 每次取一组优化器配置字典 group。 |
| **L246** | `group['lr'] = lr` | lr → 写回 group['lr']；这时只改学习率，没有改权重。 |
| **L247** | `for retry in range(8):` | range(8) → retry 依次可取 0…7，允许混合精度溢出时重算整个批次。本步 retry=0 就成功。 |
| **L248** | `opt.zero_grad(set_to_none=True)` | opt → 清空上一批或上次重试的梯度，将其置为 None。两个微批之间不会清空。 |
| **L249** | `batch_nll = 0.` | 初始化 batch_nll: float = 0.0，用来累加未归一化的损失以记日志。 |

## 微批前向、反传与参数更新

| 行 | 云端原代码 | 输入 → 输出、作用或状态变化 |
|---:|---|---|
| **L250** | `for i in range(0, len(records), MICRO):` | 4 条记录、MICRO=2 → 起始位置 i=0、2。依次处理两个微批，以降低显存占用。 |
| **L251** | `micro = records[i:i+MICRO]` | records[i:i+2] → micro: list[dict]。本题位于第二个微批，序列有效长度分别为 43、53。 |
| **L252** | `with torch.autocast('cuda', dtype=torch.float16):` | 进入 autocast → 前向采用混合精度；active_logits 返回前显式把词表分数转回 FP32。 |
| **L253** | `logits, targets = active_logits(model, micro, tok.pad_token_id)` | 两条样本 + pad ID → 第二微批输入补齐为 int64[2,53]；经过模型后返回 logits: float32[33,152064] 和 targets: int64[33]。本题占开头两行，对应 [34,151643]。 |
| **L254** | `loss_sum = F.cross_entropy(logits, targets, reduction='sum')` | logits + targets → loss_sum: 标量张量，保留梯度。第二微批是 33 项交叉熵之和；第一微批是 435 项。日志未单列两者的数值。 |
| **L255** | `scaler.scale(loss_sum/total).backward()` | loss_sum/468 → 缩放后 backward，梯度累积进 LoRA 参数的 .grad。两次各除同一个 468；尚未更新权重。本题贡献两项，不能从整步均值反推出它的 loss。 |
| **L256** | `batch_nll += loss_sum.detach().item()` | loss_sum 脱离计算图并取 Python 数值 → 累加到 batch_nll；两个微批处理完合计 1204.5901337。 |
| **L257** | `del logits, targets, loss_sum` | 删除当前微批的临时张量引用；参数 .grad 里的累计梯度仍在。 |
| **L258** | `scaler.unscale_(opt)` | opt + scaler → 将累积梯度还原到未缩放的量级，再判断是否需要裁剪。 |
| **L259** | `grad = torch.nn.utils.clip_grad_norm_(trainable, 1.)` | trainable 的梯度 + 阈值 1 → grad 为裁剪前总范数（标量张量）；超阈值才缩小梯度。本步 0.9204856，无需缩小。 |
| **L260** | `if not math.isfinite(batch_nll):` | batch_nll → 检查是否为有限数；本例正常。 |
| **L261** | `raise FloatingPointError(f'nonfinite loss on {step}')` | 如果损失为 NaN 或无穷大 → 抛异常，消息带步号；正常流程不执行。 |
| **L262** | `scale = scaler.get_scale()` | scaler 当前状态 → scale: float，记下更新前的倍率供比较；不要把它和学习率混淆。 |
| **L263** | `scaler.step(opt)` | scaler + opt → 梯度有限时执行 AdamW.step()，原地更新 LoRA 参数及优化器状态；检测到溢出时跳过。这一步才真的可能改权重。 |
| **L264** | `scaler.update()` | scaler → 根据溢出检测更新倍率，供下次使用。 |
| **L265** | `if scaler.get_scale() >= scale:` | 新倍率与旧 scale → 未下降表示此处可把优化步视为成功；下降则按 247 重试。 |
| **L266** | `break` | 成功 → break 离开 retry 循环，继续记日志。本步第一次尝试即走这里。 |
| **L267** | `else:` | 这是 for…else 的 else；只有循环耗尽且从未 break 才进入，与 265 的 if 不配对。 |
| **L268** | `raise FloatingPointError(f'AMP overflow repeated at step {step}')` | 8 次均未成功 → 抛 FloatingPointError；本步未触发。 |

## 日志、验证、保存与退出

| 行 | 云端原代码 | 输入 → 输出、作用或状态变化 |
|---:|---|---|
| **L269** | `torch.cuda.synchronize()` | CUDA 设备 → 等待已提交的 GPU 工作完成，以便准确计时。 |
| **L270** | `elapsed += time.perf_counter()-start` | elapsed + 本步墙钟时间 → 更新累计优化用时；完成第 3 步约为 2.28017 秒，不是该步单独耗时。 |
| **L271** | `entry = dict(step=step, epoch=epoch, arm=args.arm, nonpad_input_tokens=sum(r['split'] for r in records), nonpad_total_tokens=sum(len(r['ids']) for r in records), ids=[r['id'] for r in records], targets=total, nll_sum=batch_nll,` | 开始构造 entry：步号 3、轮次 1、paired、提示 token 138、有效 token 合计 606、4 个 ID、监督 468、NLL 总和 1204.5901337。 |
| **L272** | `loss=batch_nll/total, lr=lr, gradient_norm=float(grad), amp_retries=retry, train_seconds=elapsed)` | 继续填 entry：平均 loss=1204.5901337/468=2.57391054、学习率、裁剪前范数、重试数、累计耗时；字典完成。 |
| **L273** | `log_file.write(json.dumps(entry, ensure_ascii=False)+'\n')` | entry → json.dumps 成单行文字，追加换行写入 updates.jsonl；原始第 3 步可从此复核。 |
| **L274** | `if step in (1, 5) or step%100 == 0:` | step → 仅步 1、5 或 100 的倍数进入下一行；步 3 不打印 UPDATE 但已写日志。 |
| **L275** | `print('UPDATE', step, round(elapsed, 1), round(batch_nll/total, 5), flush=True)` | 选中的步号、耗时、loss → 四舍五入后打印到终端；不影响日志精度。 |
| **L276** | `if step in STAGES:` | step + STAGES → 判断是否到检查点。第 3 步不是；第 563 步首次进入。 |
| **L277** | `name = STAGES[step]` | STAGES[step] → name: str；第一次为 epoch-0.25。 |
| **L278** | `stats = validate(model, val, tok.pad_token_id)` | 当前模型 + 120 条验证集 → stats: dict，含验证 loss、逐条 NLL 等；validate 内部切至评估模式。 |
| **L279** | `save(out/f'validation-{name}.json', stats)` | stats + name → 写 validation-epoch-*.json。 |
| **L280** | `checkpoint(model, opt, scaler, out/name, dict(step=step, name=name, train_seconds=elapsed,` | 开始调用 checkpoint：传 model、opt、scaler、目标目录与状态字典（步号、名称、累计时间）；参数续至 283 行。 |
| **L281** | `total_examples=(step//563)*2250, data_hashes=data_hashes, validation_loss=stats['loss'],` | 状态字典追加累计样本数、数据指纹和验证 loss；步 563 累计 2250 条，此后按季度累计。 |
| **L282** | `peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,` | 状态字典追加截至此时 CUDA 峰值已分配显存（字节除以 2³⁰ 转 GiB）。 |
| **L283** | `peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))` | 追加峰值已预留显存（GiB），并完成调用 → 写 LoRA 权重及配置、优化器与缩放器状态、state.json；不会另存整份底座。 |
| **L284** | `model.train()` | model → 从验证模式切回训练模式，准备下一批。 |
| **L285** | `print('CHECKPOINT', name, 'validation', stats['loss'], flush=True)` | 名称 + 验证 loss → 打印 CHECKPOINT 消息；本题的第 3 步不走此行。 |
| **L286** | `final = dict(report, train_seconds=elapsed, steps=step, checkpoints=list(STAGES.values()))` | 两轮结束：report + elapsed + step + 检查点名称列表 → final: dict，含总步数 4504 和 8 个检查点。 |
| **L287** | `save(out/'completed-training.json', final)` | final + out → 写 completed-training.json，标记本组训练完成。 |
| **L288** | `print('TRAINING COMPLETE', elapsed, flush=True)` | elapsed → 打印 TRAINING COMPLETE 并刷新；函数没有显式 return，调用方得到 None。 |
