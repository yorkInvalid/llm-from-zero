"""本机网页：打一段文字立刻看到 token，并玩 5 题的猜数游戏。只监听本机，不联网。"""
import argparse
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
from learning_lab import Progress
from tokenizer_tools import load_tokenizer
from web.engine import PlayEngine

STATIC = Path(__file__).resolve().parent / 'static'
ROUTES = {'/api/measure': 'measure_text', '/api/round': 'start', '/api/guess': 'reveal', '/api/next': 'advance'}


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = 'index.html' if self.path.split('?', 1)[0] == '/' else self.path.split('?', 1)[0].lstrip('/')
            file = (STATIC / path).resolve()
            if not str(file).startswith(str(STATIC)) or not file.is_file():
                self._send(404, b'not found', 'text/plain; charset=utf-8')
                return
            kind = 'text/html' if file.suffix == '.html' else 'text/css' if file.suffix == '.css' else 'text/javascript'
            self._send(200, file.read_bytes(), f'{kind}; charset=utf-8')

        def do_POST(self):
            route = self.path.split('?', 1)[0]
            method = ROUTES.get(route)
            if method is None:
                self._send(404, b'{}', 'application/json')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                body = json.loads(self.rfile.read(length) or b'{}')
                result = getattr(engine, method)(**self._arguments(method, body))
            except ValueError as exc:
                self._json(400, {'error': str(exc)})
                return
            except (KeyError, TypeError, json.JSONDecodeError):
                self._json(400, {'error': '这一步的请求读不懂。'})
                return
            self._json(200, result)

        @staticmethod
        def _arguments(method, body):
            if method == 'measure_text':
                return {'text': body['text']}
            if method == 'start':
                return {}
            if method == 'reveal':
                return {'session_id': body['id'], 'raw_guess': body.get('guess', '')}
            return {'session_id': body['id']}

        def _json(self, status, payload):
            self._send(status, json.dumps(payload, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

        def _send(self, status, data, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            return

    return Handler


def serve(progress_path, port, rng):
    engine = PlayEngine(load_tokenizer(), Progress(progress_path), rng)
    httpd = ThreadingHTTPServer(('127.0.0.1', port), make_handler(engine))
    print(f'打开 http://127.0.0.1:{httpd.server_address[1]}')
    print('只在本机。同一份分词器，成绩写入', progress_path)
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description='课程 5 的本机试玩页')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--progress', type=Path, default=COURSE / '.learning-progress.json')
    args = parser.parse_args()
    serve(args.progress, args.port, random.Random())


if __name__ == '__main__':
    main()
