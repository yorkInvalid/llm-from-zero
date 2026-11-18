"""Blocking MCP client for this local, stateless classroom server (stdio)."""
import json
from pathlib import Path
import subprocess
import sys

SERVER = Path(__file__).with_name("inventory.py")
VERSION = "2025-06-18"


def _exchange(method, params):
    # Each example starts a fresh server, so cells may be rerun independently.
    with subprocess.Popen([sys.executable, str(SERVER)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, encoding="utf-8") as proc:
        def send(message):
            print(json.dumps({"jsonrpc": "2.0", **message}), file=proc.stdin, flush=True)

        def request(number, name, arguments):
            send({"id": number, "method": name, "params": arguments})
            for line in proc.stdout:
                reply = json.loads(line)
                if reply.get("id") == number:
                    if "error" in reply:
                        raise RuntimeError(reply["error"])
                    return reply["result"]
            raise RuntimeError("库存服务在答复之前退出了。")

        try:
            hello = request(1, "initialize", {
                "protocolVersion": VERSION, "capabilities": {},
                "clientInfo": {"name": "inventory-lesson", "version": "1.0"}})
            if hello["protocolVersion"] != VERSION:
                raise RuntimeError("库存服务不支持本课使用的协议版本。")
            send({"method": "notifications/initialized"})
            return request(2, method, params)
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


def list_tools():
    """Discover the tools registered by inventory.py."""
    return _exchange("tools/list", {})["tools"]


def call_tool(name, arguments):
    """Return the text from the teaching server's tool."""
    result = _exchange("tools/call", {"name": name, "arguments": arguments})
    if result.get("isError"):
        raise RuntimeError(result["content"])
    return result["content"][0]["text"]
