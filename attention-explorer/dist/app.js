import { compute, column } from './math.js';
import { setupRecording } from './recording.js';

const $ = id => document.getElementById(id);
const viewport = $('viewport');
const plane = $('plane');
const format = value => Math.abs(value) < 0.00005 ? '0.0000' : value.toFixed(4);
const symbol = (name, index) => `<i>${name}</i><sub>${index}</sub>`;
const steps = [
  { name: '预测任务', question: '怎样根据前三个 token 预测第四个？', text: '完整语言模型根据 x₁、x₂、x₃ 的信息预测下一个 token。Attention 在其中汇总上下文，后续网络与词表预测层据此计算候选 token 的概率。<br><br>我们跟着最后位置 x₃，计算当前头的上下文表示 b₃。输入 X = [x₁ x₂ x₃] 是 10 × 3 矩阵。比如“苹果”的含义，需要结合“我吃了一个…”或“发布新手机”等语境判断。本页的具体数值来自随机输入和初始化参数。', focus: 'X' },
  { name: '生成查询与键', question: '怎样让两个位置进行匹配？', text: '先看最后位置 x₃ 与前面的位置 x₂。两个可学习的投影分别生成 q₃ 和 k₂，让查询和被查询的位置使用适合匹配的特征。训练会调整这些投影参数。', focus: 'q' },
  { name: '比较一对', question: '怎样把匹配变成一个分数？', text: 'q₃ 与 k₂ 的对应分量逐项相乘，再相加，得到匹配分数。其他分数固定时，这个分数越大，后续分配到的注意力权重越大。', focus: 'pair' },
  { name: '比较所有位置', question: '其他位置应该怎样比较？', text: '保持 q₃ 不变，分别与 k₁、k₂、k₃ 做点积。位置 3 可以关注位置 1、2、3，包括自己。因果注意力允许查询当前位置及之前的位置，未来位置在 softmax 前被屏蔽。', focus: 'all' },
  { name: '分配权重', question: '为什么除以 √5，再做 softmax？', text: '每个头的 Q、K 都是 5 维，所以 dₖ = 5，缩放因子 √dₖ = √5。点积累加 dₖ 项，数值波动通常随维度增大。缩放控制分数尺度，避免 softmax 过早变得过于集中。然后对三个分数一起归一化，权重总和为 1。', focus: 'soft' },
  { name: '提取内容', question: '知道比例后，具体提取什么？', text: '通过 W_V 把输入转换成 value。用刚得到的三个注意力权重，对 v₁、v₂、v₃ 加权求和，得到这个头的输出 b₃。点击 b₃ 的一个分量，可以追踪它的来源。', focus: 'out' },
];
let data;
let computed;
let nodes = [];
let step = 0;
let introIndex = 0;
let expanded = null;
let selectedRow = 0;
let selectedColumn = 0;
let zoom = 1;
let offset = { x: 0, y: 0 };
let world = { width: 1600, height: 900 };

function makeNodes() {
  nodes = [
    { id: 'x1', title: symbol('x', 1), description: '10 维输入', stage: 0, lane: 0, type: 'input', index: 0 },
    { id: 'x2', title: symbol('x', 2), description: '10 维输入', stage: 0, lane: 0, type: 'input', index: 1 },
    { id: 'x3', title: symbol('x', 3), description: '10 维输入', stage: 0, lane: 0, type: 'input', index: 2 },
    { id: 'q', title: `${symbol('q', 3)} = ${symbol('W', 'Q')}${symbol('x', 3)}`, description: '5 维 query', stage: 1, lane: 1, type: 'projection', family: 'Q', index: 2 },
    { id: 'k2', title: `${symbol('k', 2)} = ${symbol('W', 'K')}${symbol('x', 2)}`, description: '5 维 key', stage: 1, lane: 1, type: 'projection', family: 'K', index: 1 },
    { id: 'k1', title: `${symbol('k', 1)} = ${symbol('W', 'K')}${symbol('x', 1)}`, description: '5 维 key', stage: 3, lane: 1, type: 'projection', family: 'K', index: 0 },
    { id: 'k3', title: `${symbol('k', 3)} = ${symbol('W', 'K')}${symbol('x', 3)}`, description: '5 维 key', stage: 3, lane: 1, type: 'projection', family: 'K', index: 2 },
    { id: 'pair', title: `${symbol('q', 3)}<sup>T</sup>${symbol('k', 2)}`, description: '一个匹配分数', stage: 2, lane: 2, type: 'pair' },
    { id: 'all', title: `${symbol('q', 3)}<sup>T</sup>[${symbol('k', 1)} ${symbol('k', 2)} ${symbol('k', 3)}]`, description: '三个匹配分数', stage: 3, lane: 3, type: 'scores' },
    { id: 'soft', title: '缩放与 softmax', description: '三个权重，总和为 1', stage: 4, lane: 4, type: 'softmax' },
    { id: 'v', title: `[${symbol('v', 1)} ${symbol('v', 2)} ${symbol('v', 3)}] = ${symbol('W', 'V')}X`, description: '5 × 3 的 value 矩阵', stage: 5, lane: 3, type: 'values' },
    { id: 'out', title: `${symbol('b', 3)} = Σ<sub>j</sub> α′<sub>3j</sub>${symbol('v', 'j')}`, description: '位置 3 的上下文表示，5 维', stage: 5, lane: 5, type: 'output' },
    { id: 'later', title: '后续 Transformer 计算', description: '流程说明', stage: 5, lane: 6, type: 'later' },
    { id: 'prediction', title: '预测第 4 个 token', description: '词表概率与生成', stage: 5, lane: 7, type: 'prediction' },
  ];
  nodes.forEach(node => { node.lane += 1; });
  nodes.unshift({ id: 'X', title: 'X = [x₁ x₂ x₃]', description: '输入矩阵，10 × 3', stage: 0, lane: 0, type: 'matrix' });
}

function table(headers, rows, caption = '') {
  return `<div class="table-scroll"><table>${caption ? `<caption>${caption}</caption>` : ''}<thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(value => `<td>${value}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function vectorTable(values, selectable = false) {
  return table(['分量', '数值'], values.map((value, index) => [
    `f${index + 1}`,
    selectable ? `<button class="numeric ${selectedRow === index ? 'selected' : ''}" data-row="${index}" aria-pressed="${selectedRow === index}">${format(value)}</button>` : format(value),
  ]));
}

function products(left, right, leftLabel, rightLabel) {
  return table(['项', leftLabel, rightLabel, '乘积'], left.map((value, index) => [
    index + 1, format(value), format(right[index]), format(value * right[index]),
  ]));
}

function weightMatrix(matrix, label) {
  return `<details><summary>查看完整的 ${label} 参数矩阵 (${matrix.length} × ${matrix[0].length})</summary>` +
    table(['输出 / 输入', ...matrix[0].map((_, index) => `f${index + 1}`)], matrix.map((row, index) => [
      `f${index + 1}`, ...row.map(value => `<span class="${index === selectedRow ? 'highlight' : ''}">${format(value)}</span>`),
    ])) + '</details>';
}

function projectionDetail(family, index) {
  const matrix = data[`W${family}`];
  const input = column(data.X, index);
  const output = computed[family][selectedRow][index];
  return `<h3>第 ${selectedRow + 1} 个分量怎么得到？</h3><p>取 W_${family} 的第 ${selectedRow + 1} 行，与 x${index + 1} 的 10 个分量逐项相乘后求和。</p>` +
    products(matrix[selectedRow], input, `W_${family} 第 ${selectedRow + 1} 行`, `x${index + 1}`) +
    `<p class="total">求和 = ${output.toFixed(7)}</p>` + weightMatrix(matrix, `W_${family}`);
}

function content(node) {
  if (node.type === 'later') return '<p>b₃ 是一个注意力头在位置 3 的输出。完整模型会合并多个头的信息，再经过输出投影、残差连接、归一化、前馈网络和更多层，得到位置 3 的最终表示。这里展示后续流程的概念说明。</p>';
  if (node.type === 'prediction') return '<p>最终表示经过语言模型的词表预测层，得到每个候选 token 的分数。对整个词表做 softmax 得到概率，再选择或采样下一个 token。本页只有随机向量与一个注意力头的计算，因此这里展示流程说明，词表概率需要完整训练后的模型。</p>';

  if (node.type === 'matrix') {
    return '<p>每列是一个输入位置，每行是一个特征分量。</p>' + table(['特征', 'x₁', 'x₂', 'x₃'], data.X.map((row, index) => [`f${index + 1}`, ...row.map(format)])) + '<h3>因果注意力允许看哪些位置？</h3>' + table(['Query / Key', '位置 1', '位置 2', '位置 3'], [['位置 1', '允许', '屏蔽', '屏蔽'], ['位置 2', '允许', '允许', '屏蔽'], ['位置 3（本页）', '允许', '允许', '允许']]);
  }
  if (node.type === 'input') {
    return `<p>这是 x${node.index + 1} 的完整 10 维向量。</p>` + vectorTable(column(data.X, node.index));
  }
  if (node.type === 'projection') {
    return `<p>点选输出分量，查看它的投影计算。</p>` + vectorTable(column(computed[node.family], node.index), true) + projectionDetail(node.family, node.index);
  }
  if (node.type === 'pair') {
    return `<p>q₃ 与 k₂ 有相同的 5 个特征维度。</p>` + products(column(computed.Q, 2), column(computed.K, 1), 'q₃', 'k₂') +
      `<p class="total">求和 = ${computed.raw[1].toFixed(7)}</p>`;
  }
  if (node.type === 'scores') {
    return `<p>选择一个 key，追踪它与 q₃ 的点积。</p>` + table(['位置', '匹配分数'], computed.raw.map((value, index) => [
      `k${index + 1}`, `<button class="numeric ${index === selectedColumn ? 'selected' : ''}" data-column="${index}">${format(value)}</button>`,
    ])) + products(column(computed.Q, 2), column(computed.K, selectedColumn), 'q₃', `k${selectedColumn + 1}`) +
      `<p class="total">求和 = ${computed.raw[selectedColumn].toFixed(7)}</p>`;
  }
  if (node.type === 'softmax') {
    return `<p>本页使用 q₃，三个输入位置都在它的可见范围内。对于 q₁ 或 q₂，未来位置的分数会在 softmax 前设为 −∞，对应权重为 0。</p><p>dₖ = 5，因此 √dₖ = √5 ≈ ${Math.sqrt(5).toFixed(6)}。</p>` +
      table(['位置', '点积', '÷ √5', '权重'], computed.raw.map((value, index) => [index + 1, format(value), format(computed.scaled[index]), format(computed.weights[index])])) +
      `<h3>softmax 的分子与分母</h3><p>为稳定数值，先从每个缩放分数减去本列最大值 ${format(computed.max)}，结果与直接归一化相同。</p>` +
      table(['位置', '缩放分数 − 最大值', 'exp（分子）'], computed.scaled.map((value, index) => [index + 1, format(value - computed.max), computed.exponentials[index].toFixed(7)])) +
      `<p class="total">分母 = 三个分子之和 = ${computed.total.toFixed(7)}</p>` +
      table(['位置', '分子 / 分母', '权重'], computed.weights.map((value, index) => [index + 1, `${format(computed.exponentials[index])} / ${format(computed.total)}`, value.toFixed(7)])) +
      `<p class="total">权重之和 = ${computed.weights.reduce((sum, value) => sum + value, 0).toFixed(7)}</p>`;
  }
  if (node.type === 'values') {
    return `<p>点击 V 的任意数值，查看它由哪个输入投影得到。</p>` + table(['特征', 'v₁', 'v₂', 'v₃'], computed.V.map((row, i) => [
      `f${i + 1}`, ...row.map((value, j) => `<button class="numeric ${i === selectedRow && j === selectedColumn ? 'selected' : ''}" data-value="${i},${j}">${format(value)}</button>`),
    ])) + projectionDetail('V', selectedColumn);
  }
  return `<p>b₃ 是当前头在位置 3 汇总得到的上下文表示。它会进入多头合并和后续网络。语言模型最终通过词表预测层产生下一个 token 的概率。</p><p>选择 b₃ 的一个分量，观察三个 value 怎样按比例贡献。</p>` + vectorTable(computed.output, true) +
    `<h3>b₃ 的第 ${selectedRow + 1} 个分量</h3>` + products(computed.weights, computed.V[selectedRow], '注意力权重', `V 的第 ${selectedRow + 1} 行`) +
    `<p class="total">加权求和 = ${computed.output[selectedRow].toFixed(7)}</p>`;
}

function applyTransform(animate = false) {
  plane.classList.toggle('moving', animate);
  plane.style.transform = `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`;
  $('zoom').textContent = `${Math.round(zoom * 100)}%`;
}

function focusNode(id, animate = true) {
  if (introIndex !== null) return;
  const node = nodes.find(item => item.id === id);
  if (!node?.element) return;
  zoom = Math.min(1, Math.max(0.55, (viewport.clientWidth - 40) / (node.width + 40)));
  if (step === 0 && !expanded) {
    const bottom = Math.max(...nodes.filter(item => item.stage === 0).map(item => item.y + item.height));
    zoom = Math.min(zoom, (viewport.clientHeight - 35) / bottom);
  }
  offset = { x: 30 - node.x * zoom, y: 12 };
  if (id === expanded) offset.y = 24 - node.y * zoom;
  applyTransform(animate);
}

function fit() {
  if (introIndex !== null) { renderIntro(); return; }
  zoom = Math.min(1, (viewport.clientWidth - 48) / world.width, (viewport.clientHeight - 48) / world.height);
  zoom = Math.max(0.12, zoom);
  offset = { x: 24, y: 24 };
  applyTransform(true);
}

function updateLayout() {
  if (introIndex !== null) return;
  const visible = nodes.filter(node => node.stage <= step);
  const maxLane = Math.max(...visible.map(node => node.lane));
  let x = 30;
  let bottom = 650;
  for (let lane = 0; lane <= maxLane; lane++) {
    const items = visible.filter(node => node.lane === lane);
    const expandedWidth = Math.min(630, Math.max(290, viewport.clientWidth - 60));
    const width = items.some(node => node.id === expanded) ? expandedWidth : 230;
    let y = 250;
    for (const node of items) {
      node.x = x;
      node.y = y;
      node.width = width;
      node.element.style.width = `${width}px`;
      node.element.style.left = `${x}px`;
      node.element.style.top = `${y}px`;
      node.height = node.element.offsetHeight;
      y += node.height + 60;
    }
    bottom = Math.max(bottom, y);
    x += width + 170;
  }
  const focus = nodes.find(node => node.id === steps[step].focus);
  const note = plane.querySelector('.teaching');
  note.style.left = `${focus.x}px`;
  note.style.width = `${Math.min(650, Math.max(Math.min(step === 0 ? 650 : 430, viewport.clientWidth - 60), focus.width))}px`;
  const noteBottom = note.offsetHeight + 30;
  // Long explanations occupy their own band above the graph.
  if (noteBottom > 220) {
    const shift = noteBottom - 220;
    visible.forEach(node => { node.y += shift; node.element.style.top = `${node.y}px`; });
    bottom += shift;
  }
  world = { width: Math.max(x, focus.x + 680), height: bottom };
  plane.style.width = `${world.width}px`;
  plane.style.height = `${world.height}px`;
  drawEdges(visible);
}

function drawEdges(visible) {
  const connections = [
    ['out', 'later', 'b₃'], ['later', 'prediction', '位置 3 的最终表示'],
    ['X', 'x1', '第 1 列'], ['X', 'x2', '第 2 列'], ['X', 'x3', '第 3 列'],
    ['x3', 'q', 'W_Q'], ['x2', 'k2', 'W_K'], ['x1', 'k1', 'W_K'], ['x3', 'k3', 'W_K'],
    ['q', 'pair', 'q₃'], ['k2', 'pair', 'k₂'], ['q', 'all', '同一个 q₃'],
    ['k1', 'all', ''], ['k2', 'all', ''], ['k3', 'all', ''], ['all', 'soft', '三个分数'],
    ['x1', 'v', ''], ['x2', 'v', ''], ['x3', 'v', 'W_V'], ['soft', 'out', '权重'], ['v', 'out', '内容'],
  ];
  const paths = [];
  for (const [from, to, label] of connections) {
    const a = visible.find(node => node.id === from);
    const b = visible.find(node => node.id === to);
    if (!a || !b) continue;
    const x1 = a.x + a.width;
    const y1 = a.y + 44;
    const x2 = b.x;
    const y2 = b.y + 44;
    // Long links pass through a dedicated band below the existing nodes.
    const distant = b.lane - a.lane > 1;
    const routeY = world.height - 35 + connections.findIndex(item => item[0] === from && item[1] === to) * 3;
    const path = distant
      ? `M ${x1} ${y1} L ${x1 + 35} ${y1} L ${x1 + 35} ${routeY} L ${x2 - 40} ${routeY} L ${x2 - 40} ${y2} L ${x2} ${y2}`
      : `M ${x1} ${y1} C ${x1 + 75} ${y1}, ${x2 - 75} ${y2}, ${x2} ${y2}`;
    paths.push(`<path d="${path}" marker-end="url(#arrow)"/><text x="${x1 + 18}" y="${y1 - 12}">${label}</text>`);
  }
  plane.querySelector('svg').outerHTML = `<svg class="connections" width="${world.width}" height="${world.height + 50}" aria-hidden="true"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8" class="arrowhead"/></marker></defs>${paths.join('')}</svg>`;
}

function saveURL() {
  const query = new URLSearchParams({ step: String(step) });
  if (expanded) query.set('node', expanded);
  query.set('row', String(selectedRow));
  query.set('col', String(selectedColumn));
  history.replaceState(null, '', `?${query}`);
}

function render() {
  if (introIndex !== null) { renderIntro(); return; }
  const activeElement = document.activeElement;
  const remembered = activeElement?.dataset?.row;
  plane.innerHTML = `<svg></svg><section class="teaching"><h2>${steps[step].question}</h2><p>${steps[step].text}</p></section>`;
  for (const node of nodes.filter(node => node.stage <= step)) {
    const element = document.createElement('section');
    element.className = `node ${node.id === steps[step].focus ? 'current' : ''} ${node.id === expanded ? 'expanded' : ''}`;
    element.dataset.id = node.id;
    element.innerHTML = `<button class="node-toggle" data-node="${node.id}" aria-expanded="${node.id === expanded}"><span class="math">${node.title}</span><span class="node-description">${node.description}</span><span class="open-label">${node.id === expanded ? '收起' : (['later', 'prediction'].includes(node.type) ? '查看流程说明' : '展开数值')}</span></button>`;
    if (node.id === expanded) element.insertAdjacentHTML('beforeend', `<div class="node-content">${content(node)}</div>`);
    node.element = element;
    plane.append(element);
  }
  $('status').textContent = `${step + 1} / ${steps.length}　${steps[step].name}`;
  $('prev').disabled = false;
  $('next').disabled = step === steps.length - 1;
  $('next').textContent = step === steps.length - 1 ? '已走完计算' : '下一步';
  $('steps').innerHTML = steps.map((item, index) => `<button data-step="${index}" ${index === step ? 'aria-current="step"' : ''}>${item.name}</button>`).join('');
  updateLayout();
  saveURL();
  if (remembered !== undefined) plane.querySelector(`[data-row="${remembered}"]`)?.focus({ preventScroll: true });
}

function changeStep(index) {
  introIndex = null;
  step = Math.max(0, Math.min(steps.length - 1, index));
  expanded = null;
  selectedRow = 0;
  selectedColumn = 0;
  render();
  focusNode(steps[step].focus);
}

function zoomAt(factor, x, y) {
  const next = Math.max(0.12, Math.min(1.8, zoom * factor));
  offset.x = x - (x - offset.x) * next / zoom;
  offset.y = y - (y - offset.y) * next / zoom;
  zoom = next;
  applyTransform();
}

function wire() {
  $('next').onclick = () => {
    if (introIndex === null) changeStep(step + 1);
    else if (introIndex < 4) { introIndex += 1; renderIntro(); }
    else changeStep(0);
  };
  $('prev').onclick = () => {
    if (introIndex !== null) { introIndex = Math.max(0, introIndex - 1); renderIntro(); }
    else if (step === 0) { introIndex = 4; renderIntro(); }
    else changeStep(step - 1);
  };
  $('reset').onclick = () => { introIndex = 0; expanded = null; renderIntro(); };
  $('fit').onclick = fit;
  $('focus').onclick = () => {
    if (introIndex !== null) { renderIntro(); return; }
    offset = { x: 30 - nodes.find(node => node.id === steps[step].focus).x * zoom, y: 12 };
    applyTransform(true);
  };
  $('plus').onclick = () => zoomAt(1.2, viewport.clientWidth / 2, viewport.clientHeight / 2);
  $('minus').onclick = () => zoomAt(1 / 1.2, viewport.clientWidth / 2, viewport.clientHeight / 2);
  $('steps').onclick = event => {
    const target = event.target.closest('[data-step]');
    if (target) changeStep(Number(target.dataset.step));
  };
  $('about-toggle').onclick = () => {
    $('about').hidden = !$('about').hidden;
    $('about-toggle').setAttribute('aria-expanded', String(!$('about').hidden));
  };
  plane.addEventListener('click', event => {
    const button = event.target.closest('button');
    if (!button || plane.classList.contains('recording-active')) return;
    if (button.dataset.node) {
      expanded = expanded === button.dataset.node ? null : button.dataset.node;
      selectedRow = 0;
      selectedColumn = 0;
      render();
      focusNode(expanded || button.dataset.node);
      plane.querySelector(`[data-node="${button.dataset.node}"]`)?.focus({ preventScroll: true });
      return;
    }
    if (button.dataset.row !== undefined) selectedRow = Number(button.dataset.row);
    if (button.dataset.column !== undefined) selectedColumn = Number(button.dataset.column);
    if (button.dataset.value) [selectedRow, selectedColumn] = button.dataset.value.split(',').map(Number);
    const savedOffset = { ...offset };
    render();
    offset = savedOffset;
    applyTransform();
  });
  plane.addEventListener('focusin', event => {
    if (event.target.matches('.node-toggle:focus-visible')) focusNode(event.target.dataset.node);
  });
  plane.addEventListener('toggle', event => {
    if (event.target.tagName === 'DETAILS') updateLayout();
  }, true);
  let drag = null;
  viewport.addEventListener('pointerdown', event => {
    if (event.target.closest('button, summary, .table-scroll, a')) return;
    drag = { x: event.clientX, y: event.clientY, startX: offset.x, startY: offset.y };
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener('pointermove', event => {
    if (!drag) return;
    offset = { x: drag.startX + event.clientX - drag.x, y: drag.startY + event.clientY - drag.y };
    applyTransform();
  });
  viewport.addEventListener('pointerup', () => { drag = null; });
  viewport.addEventListener('pointercancel', () => { drag = null; });
  viewport.addEventListener('wheel', event => {
    if (event.target.closest('.table-scroll')) return;
    event.preventDefault();
    const rect = viewport.getBoundingClientRect();
    zoomAt(Math.exp(-event.deltaY * 0.002), event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });
  viewport.addEventListener('keydown', event => {
    if (event.target !== viewport) return;
    const movement = { ArrowLeft: [60, 0], ArrowRight: [-60, 0], ArrowUp: [0, 60], ArrowDown: [0, -60] }[event.key];
    if (movement) {
      event.preventDefault();
      offset.x += movement[0]; offset.y += movement[1];
      applyTransform();
    }
  });
  new ResizeObserver(() => {
    if (introIndex !== null) renderIntro();
    else focusNode(expanded || steps[step].focus, false);
  }).observe(viewport);
}


function renderIntro() {
  const title = introIndex === 4 ? '根据前三个 token，预测下一个' : '为什么要结合上下文？';
  let body = '';
  if (introIndex < 4) {
    if (introIndex >= 1) body += `<div class="sentence"><span>我</span><span class="context ${introIndex === 3 ? 'emphasis' : ''}">吃了</span><span>一个</span><mark>苹果</mark><span>。</span><small>水果</small></div>`;
    if (introIndex >= 2) body += `<div class="sentence"><mark>苹果</mark><span class="context ${introIndex === 3 ? 'emphasis' : ''}">发布了新手机</span><span>。</span><small>公司</small></div>`;
    if (introIndex === 3) body += '<p class="intro-conclusion">结合上下文，形成更适合当前任务的表示。</p>';
  } else {
    body = '<div class="intro-flow"><span>输入 x₁、x₂、x₃</span><span aria-hidden="true">→</span><strong>Attention 汇总上下文</strong><span aria-hidden="true">→</span><span>后续网络</span><span aria-hidden="true">→</span><span>词表概率</span><span aria-hidden="true">→</span><span>生成第 4 个 token</span></div><p class="intro-conclusion">跟着最后位置 x₃，观察它怎样从已输入的位置提取信息。</p>';
  }
  plane.innerHTML = `<section class="intro-canvas"><h2>${title}</h2>${body}</section>`;
  plane.style.width = `${viewport.clientWidth}px`;
  plane.style.height = `${viewport.clientHeight}px`;
  zoom = 1;
  offset = { x: 0, y: 0 };
  applyTransform();
  $('status').textContent = `开场 ${introIndex + 1} / 5`;
  $('prev').disabled = introIndex === 0;
  $('next').disabled = false;
  $('next').textContent = introIndex === 4 ? '进入计算画布' : '下一步';
  $('steps').innerHTML = steps.map((item, index) => `<button data-step="${index}">${item.name}</button>`).join('');
  history.replaceState(null, '', `?intro=${introIndex}`);
}

async function showRecordingScene(scene) {
  if (scene.intro !== undefined) {
    introIndex = scene.intro;
    expanded = null;
    renderIntro();
    return;
  }
  introIndex = null;
  step = scene.step;
  expanded = scene.node || null;
  selectedRow = scene.row || 0;
  selectedColumn = scene.column || 0;
  render();
  if (scene.overview) { fit(); return; }
  focusNode(expanded || steps[step].focus, false);
  const container = expanded ? plane.querySelector(`[data-id="${expanded}"]`) : null;
  if (!container) return;
  let target;
  if (scene.parameters) {
    const details = container.querySelector('details');
    details.open = true;
    updateLayout();
    target = details;
  } else if (scene.section === 'last-table') {
    target = [...container.querySelectorAll('.table-scroll')].at(-1);
  } else if (scene.section) {
    target = container.querySelector(scene.section);
  }
  if (target) {
    const targetY = (target.getBoundingClientRect().top - plane.getBoundingClientRect().top) / zoom;
    offset.y = 24 - targetY * zoom;
    applyTransform();
  }
}

async function start() {
  try {
    const response = await fetch('data.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
    computed = compute(data, 2, { causal: true });
    makeNodes();
    const params = new URLSearchParams(location.search);
    introIndex = params.has('step') ? null : Math.max(0, Math.min(4, Math.floor(Number(params.get('intro'))) || 0));
    const value = Number(params.get('step'));
    step = Number.isInteger(value) ? Math.max(0, Math.min(5, value)) : 0;
    expanded = nodes.find(node => node.id === params.get('node') && node.stage <= step)?.id || null;
    selectedRow = Math.max(0, Math.min(4, Math.floor(Number(params.get('row'))) || 0));
    selectedColumn = Math.max(0, Math.min(2, Math.floor(Number(params.get('col'))) || 0));
    render();
    wire();
    focusNode(expanded || steps[step].focus, false);
    $('load').hidden = true;
    setupRecording(showRecordingScene);
    document.fonts.ready.then(() => { updateLayout(); focusNode(expanded || steps[step].focus, false); });
  } catch (error) {
    $('load').innerHTML = '数据加载失败，请通过本地 HTTP 服务打开页面。<button id="retry">重新读取</button>';
    $('retry').onclick = () => location.reload();
    console.error(error);
  }
}
start();
