"""在本机终端安全配置 TokenHub Key，输入隐藏，文件仅当前用户可读写。"""
import getpass
import os
from pathlib import Path
import tempfile
import warnings


def main():
    destination = Path(__file__).resolve().parents[1] / '.env'
    print('请粘贴腾讯云 TokenHub API Key，输入不会显示在屏幕上。')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            key = getpass.getpass('TokenHub API Key: ').strip()
    except (getpass.GetPassWarning, EOFError):
        raise SystemExit('当前终端不支持隐藏输入，请在本机 Terminal 中运行此脚本。')
    except KeyboardInterrupt:
        raise SystemExit('\n已取消。')
    if not key:
        raise SystemExit('输入为空，未保存。')
    if not key.isascii() or any(character.isspace() for character in key):
        raise SystemExit('Key 含有空格或非 ASCII 字符，请检查复制内容。')

    lines = destination.read_text().splitlines() if destination.exists() else []
    lines = [line for line in lines if line.split('=', 1)[0].strip() != 'TOKENHUB_API_KEY']
    lines.append('TOKENHUB_API_KEY=' + key)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, prefix='.env-', delete=False) as output:
            temporary = Path(output.name)
            os.fchmod(output.fileno(), 0o600)
            output.write('\n'.join(lines) + '\n')
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print('已保存 TokenHub Key，文件权限为仅当前用户可读写。')
    print('配置完成，可返回对话继续鉴权测试和音色复刻。')


if __name__ == '__main__':
    main()
