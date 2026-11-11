"""通过用户已配置的腾讯云 TokenHub 创建自有音色及试听。"""
import argparse
import base64
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'attention-explorer/output/speech'
TEXT = '今天我们一起看看，注意力机制怎样帮助模型理解上下文。Query，叫作查询；Key，叫作键；Value，提供要汇总的内容。我们用 q 三与各个 key 做点积，再除以根号五。这里的五，来自每个头的向量维度。最后按照注意力权重汇总 value，得到 b 三。'


def write_private(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=Path)
    args = parser.parse_args()
    config = {}
    for line in (ROOT / 'cloud-ai-interview/02-http-service/.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip().strip('\"').strip("'")
    local_config = {}
    local_env = ROOT / 'attention-explorer/.env'
    if local_env.exists():
        for line in local_env.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                name, value = line.split('=', 1)
                local_config[name.strip()] = value.strip().strip('"').strip("'")
    key = os.getenv('TOKENHUB_API_KEY') or local_config.get('TOKENHUB_API_KEY')
    if not key and config.get('LLM_PROVIDER') == 'tokenhub':
        if config.get('TOKENHUB_REGION', 'guangzhou') == 'guangzhou':
            key = config.get('LLM_API_KEY')
    if not key:
        raise SystemExit('缺少 TokenHub Key。请配置 TOKENHUB_API_KEY 环境变量或 attention-explorer/.env。')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT / 'voice-clone-state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print('Existing clone request state:', state.get('status'))
        print('Reuse its result or inspect the failure before retrying.')
        return
    voice_id = 'attention_voice_' + uuid.uuid4().hex[:16]
    state = {'voice_id': voice_id, 'status': 'pending', 'provider': 'tencent-tokenhub', 'preview_text': TEXT}
    write_private(state_path, state)
    payload = {
        'model': 'minimax-voice-clone',
        'voice_id': voice_id,
        'audio_data': base64.b64encode(args.audio.read_bytes()).decode('ascii'),
        'tts_model': 'minimax-speech-2.8-hd',
        'text': TEXT,
        'output_format': 'hex',
    }
    request = urllib.request.Request(
        'https://tokenhub.tencentmaas.com/v1/wand/minimax-tts/sync_clone',
        data=json.dumps(payload).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        state['status'] = 'http_error'
        state['http_status'] = error.code
        try:
            result = json.loads(error.read())
        except (ValueError, UnicodeDecodeError):
            result = {'error': 'Non-JSON HTTP error'}
        write_private(OUTPUT / 'clone-response.json', result)
        write_private(state_path, state)
        print('HTTP status:', error.code)
        print('API error:', result.get('error') or result.get('base_resp'))
        return
    except (TimeoutError, urllib.error.URLError) as error:
        state['status'] = 'uncertain'
        write_private(state_path, state)
        print('Request outcome uncertain; no automatic retry.', type(error).__name__)
        return
    write_private(OUTPUT / 'clone-response.json', result)
    base = result.get('base_resp', {})
    response_voice = result.get('voice_id') or (result.get('data') or {}).get('voice_id')
    if base.get('status_code', 0) != 0 or not response_voice:
        state['status'] = 'api_error'
        write_private(state_path, state)
        print('API error:', base or result.get('error'))
        print('Response fields:', list(result))
        return
    state.update(status='created', voice_id=response_voice)
    write_private(state_path, state)
    audio = result.get('demo_audio') or (result.get('data') or {}).get('demo_audio')
    if audio:
        if audio.startswith('https://'):
            with urllib.request.urlopen(audio, timeout=90) as response:
                audio_bytes = response.read()
        else:
            audio_bytes = bytes.fromhex(audio)
        destination = OUTPUT / 'voice-preview.mp3'
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'wb') as output:
            output.write(audio_bytes)
        state['preview'] = destination.name
        write_private(state_path, state)
        print('Preview saved:', destination)
    print('Voice creation succeeded.')
    print('Preview metadata:', result.get('extra_info', {}))


if __name__ == '__main__':
    main()
