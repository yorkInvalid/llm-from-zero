"""以整数视频帧统一每镜时长，独立补齐音轨，再合并成片。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/video'


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2500:])


def encode(clip, fps):
    clip_id = clip['id']
    video = OUTPUT / 'clips' / f'{clip_id}.mp4'
    wave = OUTPUT / 'clips' / f'{clip_id}.wav'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
         '-i', str(OUTPUT / 'frames' / clip_id / 'frames.ffconcat'),
         '-vf', f'fps={fps},format=yuv420p', '-frames:v', str(clip['frames']),
         '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'stillimage',
         '-crf', '21', '-threads', '2', '-movflags', '+faststart', str(video)])
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(OUTPUT / f'{clip_id}.mp3'),
         '-af', f'adelay={round(clip["audioLead"] * 1000)}:all=1,apad',
         '-t', str(clip['duration']), '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', str(wave)])
    return clip_id


def main():
    timeline = json.loads((OUTPUT / 'timeline.json').read_text())
    (OUTPUT / 'clips').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(encode, clip, timeline['fps']) for clip in timeline['clips']]
        for job in as_completed(jobs):
            print('Encoded', job.result(), flush=True)
    for suffix in ('mp4', 'wav'):
        lines = [f"file 'clips/{clip['id']}.{suffix}'" for clip in timeline['clips']]
        (OUTPUT / f'concat-{suffix}.txt').write_text('\n'.join(lines) + '\n')
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
         '-i', str(OUTPUT / 'concat-wav.txt'), '-af', 'loudnorm=I=-18:TP=-2:LRA=7',
         '-ar', '48000', '-ac', '1', '-c:a', 'aac', '-b:a', '160k', str(OUTPUT / 'narration.m4a')])
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
         '-i', str(OUTPUT / 'concat-mp4.txt'), '-i', str(OUTPUT / 'narration.m4a'),
         '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy', '-movflags', '+faststart',
         '-metadata', 'title=Self-attention：从三个 token 到下一个 token',
         '-metadata', 'comment=AI-generated narration using the owner-authorized cloned voice.',
         str(OUTPUT / 'attention-explained.mp4')])
    print('Video saved:', OUTPUT / 'attention-explained.mp4', flush=True)


if __name__ == '__main__':
    main()
