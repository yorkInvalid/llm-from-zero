"""真正通过本机 HTTP 调用我们的服务，不直接调用模型商。"""
import argparse
import json
import os
import sys
from time import perf_counter

import httpx


def events(lines):
    event, data = "message", []
    for line in lines:
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="便携显示器最多能借多久？续借是自动批准吗？")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--raw", action="store_true", help="显示解析后的每个 SSE 事件")
    parser.add_argument("--bad-token", action="store_true", help="故意使用错误的应用密钥，不修改模型密钥")
    args = parser.parse_args()
    key = "intentionally-wrong" if args.bad_token else os.getenv("APP_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"question": args.question, "max_tokens": args.max_tokens}
    path = "/ask/stream" if args.stream else "/ask"
    started = perf_counter()
    with httpx.Client(timeout=65, follow_redirects=False) as client:
        if not args.stream:
            response = client.post(args.url + path, headers=headers, json=payload)
            print("HTTP", response.status_code, "request_id:", response.headers.get("x-request-id"))
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
            print(f"客户端完整耗时：{perf_counter() - started:.3f} 秒")
            return 0 if response.is_success else 1
        first_text, completed = None, False
        with client.stream("POST", args.url + path, headers=headers, json=payload) as response:
            print("HTTP", response.status_code, "request_id:", response.headers.get("x-request-id"), flush=True)
            if response.status_code != 200:
                print(response.read().decode())
                return 1
            for name, data in events(response.iter_lines()):
                if args.raw:
                    print(name, json.dumps(data, ensure_ascii=False), flush=True)
                if name == "meta" and not args.raw:
                    print("来源：", data["source"], flush=True)
                elif name == "delta":
                    if first_text is None and data["text"]:
                        first_text = perf_counter() - started
                    if not args.raw:
                        print(data["text"], end="", flush=True)
                elif name == "error":
                    print("\n生成失败：", data, flush=True)
                    print("已有文字为未完成片段，不是完整答案。")
                    return 1
                elif name == "done":
                    completed = True
                    print("\n完成；usage：", data.get("usage"))
            if not completed:
                print("\n连接结束但没有 done，按未完成处理。")
                return 1
        print("首段正文耗时：", "没有正文" if first_text is None else f"{first_text:.3f} 秒")
        print(f"客户端完整耗时：{perf_counter() - started:.3f} 秒")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (httpx.HTTPError, ValueError) as exc:
        # 不打印带请求头的调试对象；这里只显示异常种类。
        print("客户端未完成请求：", type(exc).__name__)
        sys.exit(1)
