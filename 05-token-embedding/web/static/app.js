const $ = (id) => document.getElementById(id);
const COLORS = { hit: '#1a7f37', near: '#9a6700', miss: '#cf222e' };
const motion = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let session = null;

async function post(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '这一步没有完成。');
  return data;
}

function tokenStrip(blocks) {
  const box = document.createElement('div');
  box.className = 'toks';
  blocks.forEach((block, index) => {
    const chip = document.createElement('span');
    chip.className = motion ? 'tok in' : 'tok';
    chip.style.background = `hsla(${block.hue},70%,55%,.16)`;
    chip.style.borderColor = `hsla(${block.hue},60%,50%,.55)`;
    if (motion) chip.style.animationDelay = `${index * 40}ms`;
    chip.title = `ID ${block.id}`;
    chip.textContent = block.label;
    box.appendChild(chip);
  });
  return box;
}

function showTokens(target, blocks) {
  target.replaceChildren(tokenStrip(blocks));
}

function sentence(data) {
  if (!data.chars) return '打一段文字，切分马上出现。';
  const ratio = data.tokens_per_char.toFixed(2);
  const estimate = Math.round(data.per_10k_chars).toLocaleString('zh-CN');
  return `${data.chars} 个字符，${data.tokens} 个 token，每个字符约 ${ratio} 个。照这个比例，1 万个字符的同类材料约 ${estimate} 个 token。只是这一小段、这一个分词器的比例。`;
}

let pending = 0;
async function measureLive() {
  const text = $('source').value;
  const ticket = ++pending;
  $('live-error').textContent = '';
  try {
    const data = await post('/api/measure', { text });
    if (ticket !== pending) return;
    $('live-stats').textContent = sentence(data);
    showTokens($('live-tokens'), data.blocks);
  } catch (error) {
    if (ticket === pending) $('live-error').textContent = error.message;
  }
}

function scoreLine(data) {
  const best = data.best == null ? '—' : data.best;
  $('game-score').textContent = `得分 ${data.score}　最好成绩 ${best} / ${data.full}`;
}

function showQuestion(data) {
  session = data.id;
  scoreLine(data);
  $('game-title').textContent = `第 ${data.index + 1} / ${data.total} 题 · ${data.category}`;
  $('game-text').textContent = data.text;
  $('game-hint').textContent = `这段有 ${data.chars} 个字符。先报一个 token 数。`;
  $('game-tokens').replaceChildren();
  $('game-ruler').hidden = true;
  $('game-result').textContent = '';
  $('game-error').textContent = '';
  $('summary').hidden = true;
  $('guess').hidden = false;
  $('guess').disabled = false;
  $('guess').value = '';
  $('guess-form').querySelector('button[type="submit"]').hidden = false;
  $('next').hidden = true;
  $('again').hidden = true;
  $('guess').focus();
}

function drawRuler(guess, actual, verdict) {
  const ruler = $('game-ruler');
  ruler.hidden = false;
  ruler.replaceChildren();
  const top = Math.max(guess, actual) * 1.25 + 1;
  const at = (value) => `${value / top * 100}%`;
  const color = COLORS[verdict];
  const low = Math.min(guess, actual) / top * 100;
  const high = Math.max(guess, actual) / top * 100;
  const line = document.createElement('div');
  line.className = 'line';
  const gap = document.createElement('div');
  gap.className = 'gap';
  gap.style.left = `${low}%`;
  gap.style.width = `${high - low}%`;
  gap.style.background = color;
  ruler.append(line, gap, mark('up', at(guess), `你猜 ${guess}`, color), mark('down', at(actual), `真实 ${actual}`), mark('down', '0%', '0', '#57606a'));
}

function mark(place, left, text, color) {
  const node = document.createElement('div');
  node.className = `mark ${place}`;
  node.style.left = left;
  if (color) node.style.color = color;
  const label = document.createElement('span');
  label.textContent = text;
  const stem = document.createElement('i');
  if (place === 'up') node.append(label, stem);
  else node.append(stem, label);
  return node;
}

function showReveal(data) {
  scoreLine(data);
  showTokens($('game-tokens'), data.blocks);
  drawRuler(data.guess, data.tokens, data.verdict);
  const result = document.createElement('span');
  result.className = `big ${data.verdict}`;
  result.textContent = data.words;
  const detail = document.createElement('span');
  detail.textContent = `　真实 ${data.tokens} 个 token，你猜 ${data.guess}。${data.streak >= 2 ? `　连击 ×${data.streak}` : ''}`;
  $('game-result').replaceChildren(result, detail);
  const extra = document.createElement('p');
  extra.className = 'muted';
  extra.textContent = sentence(data);
  $('game-result').append(extra);
  $('guess').disabled = true;
  $('guess-form').querySelector('button[type="submit"]').hidden = true;
  $('next').hidden = false;
  $('next').textContent = data.last ? '看总成绩' : '下一题';
  $('next').focus();
}

function showSummary(data) {
  scoreLine({ score: data.score, best: data.best, full: data.full });
  $('game-title').textContent = '一轮结束';
  $('game-text').textContent = '';
  $('game-hint').textContent = '';
  $('game-tokens').replaceChildren();
  $('game-ruler').hidden = true;
  $('guess').hidden = true;
  $('guess-form').querySelector('button[type="submit"]').hidden = true;
  $('next').hidden = true;
  $('again').hidden = false;
  const summary = $('summary');
  summary.hidden = false;
  summary.replaceChildren();
  const headline = document.createElement('p');
  const score = document.createElement('span');
  score.className = 'big';
  score.textContent = `${data.score}`;
  headline.append(score, document.createTextNode(` / ${data.full} 分　命中 ${data.hits} 题　最长连击 ${data.best_streak}`));
  if (data.new_best) {
    const record = document.createElement('span');
    record.className = 'hit';
    record.textContent = '　新纪录！';
    headline.append(record);
  }
  summary.append(headline);
  data.lines.forEach((line) => {
    const paragraph = document.createElement('p');
    paragraph.textContent = line;
    summary.append(paragraph);
  });
  summary.append(chart(data.rows));
  const closing = document.createElement('p');
  closing.className = 'muted';
  closing.textContent = '同样多的字符，token 数可以差好几倍。给客户估算时按材料类型分别抽样，拿上面那格贴一段真实材料看看。';
  summary.append(closing);
}

function chart(rows) {
  const top = Math.max(...rows.map((row) => Math.max(row.actual, row.guess) / row.chars)) * 1.1;
  const grid = document.createElement('div');
  grid.className = 'chart';
  const caption = document.createElement('p');
  caption.className = 'muted';
  caption.textContent = '每字符多少 token：色条是真实值，竖线是你猜的。';
  rows.forEach((row) => {
    const name = document.createElement('span');
    name.textContent = row.category;
    const track = document.createElement('div');
    track.className = 'track';
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = `${row.actual / row.chars / top * 100}%`;
    bar.style.background = COLORS[row.verdict];
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = `${row.guess / row.chars / top * 100}%`;
    track.append(bar, tick);
    const ratio = document.createElement('span');
    ratio.className = 'muted';
    ratio.textContent = row.ratio.toFixed(2);
    grid.append(name, track, ratio);
  });
  const wrap = document.createElement('div');
  wrap.append(caption, grid);
  return wrap;
}

$('source').addEventListener('input', () => {
  clearTimeout($('source')._timer);
  $('source')._timer = setTimeout(measureLive, 180);
});

$('guess-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('game-error').textContent = '';
  try {
    showReveal(await post('/api/guess', { id: session, guess: $('guess').value }));
  } catch (error) {
    $('game-error').textContent = error.message;
  }
});

$('next').addEventListener('click', async () => {
  try {
    const data = await post('/api/next', { id: session });
    if (data.phase === 'done') showSummary(data);
    else showQuestion(data);
  } catch (error) {
    $('game-error').textContent = error.message;
  }
});

$('again').addEventListener('click', () => post('/api/round', {}).then(showQuestion));

$('source').value = '乙方应于每季度结束后十个工作日内提交服务报告。';
measureLive();
post('/api/round', {}).then(showQuestion).catch((error) => {
  $('game-error').textContent = error.message;
});
