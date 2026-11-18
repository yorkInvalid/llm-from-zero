"""One question, at most one read-only tool call and two model generations."""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def load_generator(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, local_files_only=True).to(device).eval()
    print("device:", device, flush=True)
    @torch.inference_mode()
    def generate(messages, tools):
        inputs = tok.apply_chat_template(messages, tools=tools, enable_thinking=False,
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=96, do_sample=False,
                                pad_token_id=tok.eos_token_id)
        new_ids = output[0, inputs.input_ids.shape[1]:]
        print(f"tokens: input={inputs.input_ids.shape[1]} output={len(new_ids)}")
        return tok.decode(new_ids, skip_special_tokens=True).strip()
    return generate

async def main(args):
    generate = None if args.probe else load_generator(args.model)
    server = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name("inventory.py"))])
    async with stdio_client(server) as (reader, writer), ClientSession(reader, writer) as client:
        await client.initialize()
        available = (await client.list_tools()).tools
        tools = [{"type":"function", "function":{"name":t.name, "description":t.description,
                  "parameters":t.inputSchema}} for t in available]
        print("1. MCP → 可用工具:", json.dumps(tools, ensure_ascii=False))
        if args.probe:
            print("直接调用（没有模型）:", (await client.call_tool("get_stock", {"sku":"P28"})).content[0].text)
            return
        msgs = [{"role":"system","content":"你是库存助手。查询库存必须调用工具，不要猜测。收到工具结果后直接用中文简短回答。其他问题直接回答。"},
                {"role":"user","content":args.question}]
        raw = generate(msgs, tools)
        print("2. 模型首次输出:", raw)
        if "<tool_call>" not in raw and "</tool_call>" not in raw:
            print("未调用工具。")
            return
        match = re.fullmatch(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.S)
        if not match:
            raise ValueError("调用格式不正确；未执行工具。")
        call = json.loads(match[1])
        if not isinstance(call, dict) or call.get("name") != "get_stock":
            raise ValueError("未授权的工具名。")
        params = call.get("arguments")
        if not isinstance(params, dict) or set(params) != {"sku"} or not isinstance(params["sku"], str) or not re.fullmatch(r"[A-Za-z0-9-]{1,32}", params["sku"]):
            raise ValueError("SKU 参数不合法；未执行工具。")
        result = await client.call_tool(call["name"], params)
        if result.isError:
            raise RuntimeError("库存工具返回错误。")
        value = result.content[0].text
        print("3. MCP → 工具实际回执:", value)
        msgs += [{"role":"assistant","content":"","tool_calls":[{"type":"function","function":call}]},
                 {"role":"tool","content":value}]
        final = generate(msgs, tools)
        if "<tool_call>" in final:
            raise ValueError("已达到一轮工具调用上限。")
        print("4. 模型最终回答:", final)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="weights")
    p.add_argument("--probe", action="store_true")
    p.add_argument("question", nargs="?", default="请查一下商品 P28 现在有多少件库存？")
    asyncio.run(main(p.parse_args()))
