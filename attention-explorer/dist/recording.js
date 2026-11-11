const clock = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;

export async function setupRecording(showScene) {
  const toggle = document.getElementById('record-toggle');
  const bar = document.getElementById('recording');
  const select = document.getElementById('cue-select');
  let scenes;
  let index = 0;
  let active = false;

  async function display() {
    const cue = scenes[index];
    delete bar.dataset.cueReady;
    select.value = String(index);
    document.getElementById('cue-time').textContent = `目标时间 ${clock(cue.start)}–${clock(cue.end)}，切换后静默 3 秒`;
    document.getElementById('cue-pointer').textContent = cue.pointer;
    document.getElementById('cue-speech').textContent = cue.speech;
    document.getElementById('cue-prev').disabled = index === 0;
    document.getElementById('cue-next').disabled = index === scenes.length - 1;
    await new Promise(resolve => requestAnimationFrame(resolve));
    await showScene(cue.scene);
    bar.dataset.cueReady = String(index);
  }

  toggle.onclick = async () => {
    if (!scenes) {
      toggle.disabled = true;
      toggle.textContent = '正在读取分镜';
      try {
        const response = await fetch('recording.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        scenes = await response.json();
        select.innerHTML = scenes.map((cue, i) => `<option value="${i}">${i + 1}. ${cue.title}</option>`).join('');
      } catch (error) {
        toggle.textContent = '分镜读取失败，点击重试';
        toggle.disabled = false;
        return;
      }
      toggle.disabled = false;
    }
    active = !active;
    toggle.textContent = active ? '退出按稿演示' : '按稿演示';
    toggle.setAttribute('aria-pressed', String(active));
    bar.hidden = !active;
    document.body.classList.toggle('recording-mode', active);
    document.querySelector('.progress').hidden = active;
    document.getElementById('steps').hidden = active;
    // Manual camera inspection stays available. Structural navigation follows the cue.
    document.getElementById('reset').disabled = active;
    document.getElementById('plane').classList.toggle('recording-active', active);
    if (active) {
      index = 0;
      await display();
    }
  };
  select.onchange = async () => { index = Number(select.value); await display(); };
  document.getElementById('cue-prev').onclick = async () => { index = Math.max(0, index - 1); await display(); };
  document.getElementById('cue-next').onclick = async () => { index = Math.min(scenes.length - 1, index + 1); await display(); };
}
