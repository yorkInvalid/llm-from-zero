/** Capture deterministic browser scenes and burn provider-timed captions into frames. */
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'output/video');
const puppeteerPath = process.env.PUPPETEER_MODULE;
if (!puppeteerPath) throw new Error('Set PUPPETEER_MODULE to the installed puppeteer-core module path.');
const { default: puppeteer } = await import(pathToFileURL(puppeteerPath));
const cues = JSON.parse(await readFile(path.join(root, 'dist/recording.json'), 'utf8'));
const manifest = JSON.parse(await readFile(path.join(output, 'manifest.json'), 'utf8'));
const before = 0.8;
const after = 0.7;
const framesPerSecond = 25;
const timeline = [];
let totalFrames = 0;
let srtCounter = 1;
let srt = '';

function timestamp(seconds) {
  const milliseconds = Math.round(seconds * 1000);
  const hours = Math.floor(milliseconds / 3600000);
  const minutes = Math.floor(milliseconds / 60000) % 60;
  const wholeSeconds = Math.floor(milliseconds / 1000) % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(wholeSeconds).padStart(2, '0')},${String(milliseconds % 1000).padStart(3, '0')}`;
}

const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 1 });
  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle2' });
  await page.waitForSelector('.intro-canvas');
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: `
    body { height: 900px; overflow: hidden; }
    header.top { padding: 16px 30px; }
    .tools, footer, #load { display: none !important; }
    #viewport, .recording-mode #viewport { height: 650px; min-height: 0; border-bottom: 0; }
    #plane.moving { transition: none !important; }
    .node-toggle .open-label { display: none; }
    .intro-canvas { max-width: 1200px; padding: 46px 70px; }
    .intro-canvas h2 { font-size: 34px; }
    .sentence { font-size: 34px; }
    #video-subtitle { position: fixed; bottom: 18px; left: 90px; right: 90px; min-height: 75px; display: flex; align-items: center; justify-content: center; text-align: center; font-size: 25px; line-height: 1.65; padding: 12px 24px; background: var(--paper); border-top: 1px solid var(--rule); z-index: 1000; }
    .video-emphasis { outline: 2px solid var(--signal); outline-offset: 4px; }
    #video-label { position: fixed; top: 22px; right: 30px; font-size: 14px; color: var(--muted); }
  ` });
  // Start recording mode before hiding its native controls.
  await page.evaluate(() => document.getElementById('record-toggle').click());
  await page.waitForSelector('#cue-select option');
  await page.evaluate(() => {
    const subtitle = document.createElement('div'); subtitle.id = 'video-subtitle'; document.body.append(subtitle);
    const label = document.createElement('div'); label.id = 'video-label'; document.body.append(label);
  });

  for (let index = 0; index < cues.length; index++) {
    const id = String(index + 1).padStart(2, '0');
    const clip = manifest.clips[id];
    if (clip?.status !== 'complete') throw new Error(`Clip ${id} is not complete.`);
    const directory = path.join(output, 'frames', id);
    await mkdir(directory, { recursive: true });
    const frameCount = Math.ceil((before + clip.duration + after) * framesPerSecond);
    const duration = frameCount / framesPerSecond;
    const clipStart = totalFrames / framesPerSecond;
    const subtitles = JSON.parse(await readFile(path.join(output, clip.subtitles), 'utf8'));
    const segments = [];
    let cursor = 0;
    for (const sub of subtitles) {
      const start = Math.max(cursor, Math.min(duration, before + sub.time_begin / 1000));
      const end = Math.max(start, Math.min(before + clip.duration, before + sub.time_end / 1000));
      if (start > cursor) segments.push({ start: cursor, end: start, text: '' });
      if (end > start) {
        segments.push({ start, end, text: sub.text });
        srt += `${srtCounter++}\n${timestamp(clipStart + start)} --> ${timestamp(clipStart + end)}\n${sub.text}\n\n`;
      }
      cursor = end;
    }
    if (cursor < duration) segments.push({ start: cursor, end: duration, text: '' });
    await page.evaluate(value => {
      const select = document.getElementById('cue-select');
      select.value = String(value); select.dispatchEvent(new Event('change', { bubbles: true }));
    }, index);
    await page.waitForFunction(value => document.getElementById('recording').dataset.cueReady === String(value), {}, index);
    await page.evaluate(title => { document.getElementById('video-label').textContent = `${title}　｜　AI 合成配音`; }, `${index + 1} / 31　${cues[index].title}`);
    // Auto-fit the relevant calculation block vertically when it is taller than the visible stage.
    await page.evaluate(scene => {
      if (!scene.node) return;
      const node = document.querySelector(`[data-id="${scene.node}"]`);
      const plane = document.getElementById('plane');
      const viewport = document.getElementById('viewport');
      let target = node;
      if (scene.parameters) target = node.querySelector('details');
      else if (scene.section === 'last-table') target = [...node.querySelectorAll('.table-scroll')].at(-1);
      else if (scene.section) target = node.querySelector(scene.section);
      const matrix = new DOMMatrix(getComputedStyle(plane).transform);
      const rect = target.getBoundingClientRect();
      const planeRect = plane.getBoundingClientRect();
      const x = (node.getBoundingClientRect().left - planeRect.left) / matrix.a;
      const y = (rect.top - planeRect.top) / matrix.a;
      let height = target.offsetHeight;
      if (scene.section === 'h3') {
        const nextTable = target.nextElementSibling?.matches('.table-scroll') ? target.nextElementSibling : target.nextElementSibling?.nextElementSibling;
        if (nextTable) height = (nextTable.getBoundingClientRect().bottom - rect.top) / matrix.a + 60;
      }
      // Full projection details are covered by later close-up shots; keep its output legible here.
      if (!scene.section && !scene.parameters && ['q', 'k2', 'v', 'X', 'out', 'soft', 'all'].includes(scene.node)) {
        const firstTable = node.querySelector('.table-scroll');
        height = (firstTable.getBoundingClientRect().bottom - rect.top) / matrix.a + 20;
      }
      const scale = Math.min(1.15, (viewport.clientHeight - 45) / Math.max(height, 220));
      plane.style.transition = 'none';
      plane.style.transform = `translate(${80 - x * scale}px, ${20 - y * scale}px) scale(${scale})`;
    }, cues[index].scene);
    await page.evaluate(cueIndex => {
      document.querySelectorAll('.video-emphasis').forEach(el => el.classList.remove('video-emphasis'));
      const node = document.querySelector('.node.expanded');
      let target;
      if (cueIndex === 1) target = document.querySelector('.sentence mark');
      else if (cueIndex === 2) target = document.querySelectorAll('.sentence mark')[1];
      else if ([9, 10, 16, 17, 24].includes(cueIndex)) target = node?.querySelector('h3');
      else if (cueIndex === 11) target = node?.querySelector('details');
      else if ([13, 26].includes(cueIndex)) target = node?.querySelector('.total');
      else if (cueIndex === 21) target = node?.querySelector('.total');
      else if (cueIndex === 22) target = [...(node?.querySelectorAll('.table-scroll') || [])].at(-1);
      target?.classList.add('video-emphasis');
    }, index);
    const concatLines = ['ffconcat version 1.0'];
    for (let part = 0; part < segments.length; part++) {
      const segment = segments[part];
      await page.evaluate(text => { document.getElementById('video-subtitle').textContent = text; }, segment.text);
      const file = `${String(part).padStart(3, '0')}.png`;
      await page.screenshot({ path: path.join(directory, file) });
      concatLines.push(`file '${file}'`, `duration ${(segment.end - segment.start).toFixed(8)}`);
    }
    concatLines.push(`file '${String(segments.length - 1).padStart(3, '0')}.png'`);
    await writeFile(path.join(directory, 'frames.ffconcat'), concatLines.join('\n') + '\n');
    timeline.push({ id, title: clip.title, start: clipStart, duration, frames: frameCount, audioDuration: clip.duration, audioLead: before, subtitles: subtitles.length });
    totalFrames += frameCount;
    console.log(`Captured ${id}/31: ${segments.length} subtitle states, ${duration.toFixed(2)}s`);
  }
  if (errors.length) throw new Error(errors.join('\n'));
  await writeFile(path.join(output, 'timeline.json'), JSON.stringify({ fps: framesPerSecond, totalFrames, duration: totalFrames / framesPerSecond, clips: timeline }, null, 2));
  await writeFile(path.join(output, 'attention.srt'), srt);
} finally {
  await browser.close();
}
