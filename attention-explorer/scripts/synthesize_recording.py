"""按分镜生成腾讯云 TokenHub 配音，保存字幕和真实时长，支持断点继续。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/video'
MODEL = 'minimax-speech-2.8-hd'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_key():
    if os.getenv('TOKENHUB_API_KEY'):
        return os.environ['TOKENHUB_API_KEY']
    for line in (ROOT / '.env').read_text().splitlines():
        name, separator, value = line.partition('=')
        if separator and name.strip() == 'TOKENHUB_API_KEY':
            return value.strip().strip('"').strip("'")
    raise SystemExit('Missing TOKENHUB_API_KEY')


def duration(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(path)], capture_output=True, text=True, check=True)
    return float(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=31)
    args = parser.parse_args()
    scenes = json.loads((ROOT / 'dist/recording.json').read_text())
    voice = json.loads((ROOT / 'output/speech/voice-clone-state.json').read_text())['voice_id']
    key = load_key()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'model': MODEL, 'voice_id': voice, 'clips': {}}
    if manifest['voice_id'] != voice:
        raise SystemExit('Voice changed; review existing clips before continuing.')
    for index, cue in enumerate(scenes[:args.limit]):
        clip_id = f'{index + 1:02d}'
        digest = hashlib.sha256(cue['speech'].encode()).hexdigest()
        prior = manifest['clips'].get(clip_id)
        if prior:
            if prior['text_sha256'] != digest:
                raise SystemExit(f'Clip {clip_id} script changed; review before replacement.')
            if prior['status'] == 'complete':
                print(f'{clip_id} cached', flush=True)
                continue
            raise SystemExit(f'Clip {clip_id} has status {prior["status"]}; inspect before retrying.')
        entry = {'status': 'pending', 'text_sha256': digest, 'title': cue['title'], 'speech': cue['speech'], 'scene': cue['scene']}
        manifest['clips'][clip_id] = entry
        write_json(manifest_path, manifest)
        payload = {
            'model': MODEL, 'text': cue['speech'],
            'voice_setting': {'voice_id': voice, 'speed': 1.0, 'vol': 1, 'pitch': 0},
            'audio_setting': {'sample_rate': 32000, 'bitrate': 128000, 'format': 'mp3', 'channel': 1},
            'subtitle_enable': True, 'subtitle_type': 'sentence', 'output_format': 'hex', 'stream': False,
        }
        request = urllib.request.Request('https://tokenhub.tencentmaas.com/v1/wand/minimax-tts/sync_tts', data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            entry['status'] = 'http_error'
            entry['http_status'] = error.code
            write_json(OUTPUT / f'{clip_id}-error.json', json.loads(error.read()))
            write_json(manifest_path, manifest)
            raise SystemExit(f'HTTP {error.code} for clip {clip_id}')
        except (urllib.error.URLError, TimeoutError):
            entry['status'] = 'uncertain'
            write_json(manifest_path, manifest)
            raise SystemExit(f'Uncertain result for {clip_id}; stopped without automatic retry.')
        if result.get('base_resp', {}).get('status_code', 0) != 0 or result.get('error'):
            entry['status'] = 'api_error'
            write_json(OUTPUT / f'{clip_id}-error.json', result)
            write_json(manifest_path, manifest)
            raise SystemExit(f'API error: {result.get("base_resp") or result.get("error")}')
        response_data = result.get('data') or {}
        audio = response_data.get('audio')
        if not audio:
            raise SystemExit(f'Clip {clip_id} missing audio')
        if audio.startswith('https://'):
            with urllib.request.urlopen(audio, timeout=90) as response:
                audio_bytes = response.read()
        else:
            audio_bytes = bytes.fromhex(audio)
        audio_path = OUTPUT / f'{clip_id}.mp3'
        audio_path.write_bytes(audio_bytes)
        os.chmod(audio_path, 0o600)
        metadata = {key: value for key, value in result.items() if key != 'data'}
        metadata['data'] = {key: value for key, value in response_data.items() if key != 'audio'}
        write_json(OUTPUT / f'{clip_id}-response.json', metadata)
        subtitle_url = response_data.get('subtitle_file')
        if subtitle_url:
            with urllib.request.urlopen(subtitle_url, timeout=90) as response:
                subtitles = json.load(response)
            write_json(OUTPUT / f'{clip_id}-subtitles.json', subtitles)
        entry.update(status='complete', audio=audio_path.name, duration=duration(audio_path), usage=result.get('usage'), subtitles=f'{clip_id}-subtitles.json' if subtitle_url else None)
        write_json(manifest_path, manifest)
        print(f'{clip_id}/31 complete, {entry["duration"]:.2f}s, subtitles={bool(subtitle_url)}', flush=True)


if __name__ == '__main__':
    main()
