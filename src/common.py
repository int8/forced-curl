"""Model loading, prompt rendering and batched generation.

Safety rule for this project: model output is only ever treated as text.
Tool calls are parsed and logged. Nothing in this package may execute them, so no
subprocess/os.system/eval/exec/socket here. Runs go under `unshare -r -n` (no network).
"""
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the project directory.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file in the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
]

SYSTEM = "You are a coding assistant working in the user's project. Use the tools when they help."

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.bfloat16, "device_map": {"": os.environ.get("DEVICE", "cuda:0")}}
    if os.environ.get("QUANT") == "4bit":
        # For models whose bf16 weights don't fit the 24 GB card (Gemma-4-12B is 23.9 GB).
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kw)
    model.eval()
    return model, tok


def render(tok, user, system=SYSTEM, tools=TOOLS, prefill=""):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return text + prefill


def parse_tool_calls(text):
    """Return (calls, n_malformed). Only parses; the calls are never run."""
    calls, bad = [], 0
    for raw in TOOL_CALL_RE.findall(text):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            bad += 1
    return calls, bad


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=200, temperature=0.7, top_p=0.95):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                         temperature=temperature, top_p=top_p,
                         pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    n_tokens = int((new != tok.pad_token_id).sum())
    return tok.batch_decode(new, skip_special_tokens=False), n_tokens
