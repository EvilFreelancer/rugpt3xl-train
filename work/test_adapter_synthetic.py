#!/usr/bin/env python3
"""
Synthetic smoke-test for the ruGPT3XL LoRA adapter.

What this script checks:
- reasoning in Russian and English
- tool call generation in Russian and English
- follow-up answer after a synthetic tool result
- Markdown report with full input and output for each case

Usage example:
    python work/test_adapter_synthetic.py \
        --adapter-path ./work/runs/rugpt3xl-8k-fsdp/lora_adapter_final
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


GIGACHAT3_SPECIAL_TOKENS = [
    "<|role_sep|>",
    "<|message_sep|>",
    "<|function_call|>",
    "<|file|>",
    "<|/file|>",
]

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression with + - * / and parentheses",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expression to evaluate, for example (17*23)+19",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

SYNTHETIC_CASES = [
    {
        "id": "reasoning_ru_inventory",
        "type": "reasoning",
        "language": "ru",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Отвечай на русском. Покажи краткое рассуждение по шагам и затем итог."
                ),
            },
            {
                "role": "user",
                "content": (
                    "На складе было 120 деталей. Утром отправили 35, днем привезли 18, "
                    "вечером отправили 27. Сколько деталей осталось"
                ),
            },
        ],
        "expected_answer_regex": r"\b76\b",
    },
    {
        "id": "reasoning_en_tickets",
        "type": "reasoning",
        "language": "en",
        "messages": [
            {
                "role": "system",
                "content": "Answer in English. Show short step-by-step reasoning and final result.",
            },
            {
                "role": "user",
                "content": (
                    "A team solved 48 tickets on Monday, 37 on Tuesday and 29 on Wednesday, "
                    "then 26 new tickets arrived. What is the net number of solved tickets"
                ),
            },
        ],
        "expected_answer_regex": r"\b88\b",
    },
    {
        "id": "tool_call_ru_weather",
        "type": "tool_call",
        "language": "ru",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Отвечай на русском. Если нужны внешние данные, сначала вызови функцию. "
                    "Ничего не придумывай."
                ),
            },
            {
                "role": "user",
                "content": "Какая сейчас погода в Санкт-Петербурге. Нужна температура в Цельсиях.",
            },
        ],
        "tools": DEFAULT_TOOLS,
        "expected_tool_name": "get_weather",
        "required_tool_args": ["city"],
        "expected_final_answer_regex": r"\b7\b",
    },
    {
        "id": "tool_call_en_calculator",
        "type": "tool_call",
        "language": "en",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer in English. If calculation is requested, call the right tool first."
                ),
            },
            {
                "role": "user",
                "content": "Please calculate (17*23)+19 using a tool, then give the final answer.",
            },
        ],
        "tools": DEFAULT_TOOLS,
        "expected_tool_name": "calculator",
        "required_tool_args": ["expression"],
        "expected_final_answer_regex": r"\b410\b",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic adapter eval")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("./work/runs/rugpt3xl-8k-fsdp/lora_adapter_final"),
        help="Path to extracted LoRA adapter directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model path or HF id",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Model device_map passed to transformers",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load base model in 4-bit NF4 for lower VRAM",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--context-window",
        type=int,
        default=8192,
        help="Maximum total tokens for prompt plus generation",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=8,
        help="Prevent repeating n-grams during generation, set 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to save Markdown report. Defaults to <adapter>/synthetic_eval_report.md",
    )
    parser.add_argument(
        "--json-output-path",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_adapter_weight_file(adapter_path: Path) -> Path | None:
    for file_name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_path / file_name
        if candidate.exists():
            return candidate
    return None


def patch_tokenizer_runtime(tokenizer: Any, template_path: Path) -> dict[str, Any]:
    changes = {"added_tokens": [], "template_overridden": False}

    unk_id = tokenizer.unk_token_id
    missing_tokens = []
    for token in GIGACHAT3_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == unk_id:
            missing_tokens.append(token)

    if missing_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
        changes["added_tokens"] = missing_tokens

    current_template = tokenizer.chat_template or ""
    if "<|role_sep|>" not in current_template and template_path.exists():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8").strip()
        changes["template_overridden"] = True

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return changes


def build_model(
    adapter_path: Path,
    base_model_override: str | None,
    device_map: str,
    load_in_4bit: bool,
) -> tuple[Any, Any, str, dict[str, Any]]:
    adapter_config = read_json(adapter_path / "adapter_config.json")
    base_model_name = base_model_override or adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    template_path = Path(__file__).resolve().with_name("chat_template_gigachat3.jinja")
    tokenizer_patch = patch_tokenizer_runtime(tokenizer, template_path)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["device_map"] = device_map
    else:
        model_kwargs["dtype"] = torch.float32
        model_kwargs["device_map"] = None

    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-in-4bit requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["dtype"] = None

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    model.eval()

    return model, tokenizer, base_model_name, tokenizer_patch


def get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    if tools:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            pass

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_new_tokens: int,
    context_window: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[str, dict[str, Any]]:
    prompt = apply_chat_template(tokenizer, messages, tools)
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    prompt_tokens_total = int(input_ids.shape[-1])
    truncated_prompt_tokens = 0
    min_prompt_tokens = 16
    if prompt_tokens_total >= context_window:
        keep_tokens = max(min_prompt_tokens, context_window - 1)
        truncated_prompt_tokens = prompt_tokens_total - keep_tokens
        input_ids = input_ids[:, -keep_tokens:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -keep_tokens:]

    prompt_tokens_used = int(input_ids.shape[-1])
    available_new_tokens = max(1, context_window - prompt_tokens_used)
    effective_max_new_tokens = min(max_new_tokens, available_new_tokens)

    max_token_id = int(input_ids.max().item())
    embedding_rows = int(model.get_input_embeddings().weight.shape[0])
    if max_token_id >= embedding_rows:
        raise RuntimeError(
            f"input token id {max_token_id} is out of embedding range {embedding_rows - 1}"
        )

    inputs: dict[str, Any] = {
        "input_ids": input_ids.to(device),
    }
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask.to(device)

    stop_token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(int(tokenizer.eos_token_id))
    message_sep_id = tokenizer.convert_tokens_to_ids("<|message_sep|>")
    if message_sep_id is not None and message_sep_id != tokenizer.unk_token_id:
        stop_token_ids.append(int(message_sep_id))
    stop_token_ids = list(dict.fromkeys(stop_token_ids))
    if not stop_token_ids:
        raise RuntimeError("Could not determine any stop token ids")

    do_sample = temperature > 0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": effective_max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": stop_token_ids if len(stop_token_ids) > 1 else stop_token_ids[0],
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        gen_kwargs["renormalize_logits"] = True
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
    stopped_on_message_sep = False
    if message_sep_id is not None and message_sep_id != tokenizer.unk_token_id:
        stopped_on_message_sep = bool((new_tokens == int(message_sep_id)).any().item())
    generation_info = {
        "context_window": context_window,
        "requested_max_new_tokens": max_new_tokens,
        "effective_max_new_tokens": effective_max_new_tokens,
        "prompt_tokens_total": prompt_tokens_total,
        "prompt_tokens_used": prompt_tokens_used,
        "truncated_prompt_tokens": truncated_prompt_tokens,
        "max_input_token_id": max_token_id,
        "stop_token_ids": stop_token_ids,
        "stopped_on_message_sep": stopped_on_message_sep,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    return decoded, generation_info


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_tool_call(raw_text: str) -> tuple[str | None, dict[str, Any], str | None]:
    if "<|function_call|>" not in raw_text:
        return None, {}, "missing <|function_call|> marker"

    payload = raw_text.split("<|function_call|>", 1)[1]
    payload = payload.split("<|message_sep|>", 1)[0].strip()

    json_blob = extract_first_json_object(payload)
    if not json_blob:
        return None, {}, "tool call JSON not found"

    try:
        obj = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        return None, {}, f"invalid JSON in tool call: {exc}"

    tool_name = obj.get("name")
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw_arguments": args}
    if not isinstance(args, dict):
        args = {"value": args}

    return tool_name, args, None


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("<|message_sep|>", "\n")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|function_call|>", "")
    return text.strip()


def language_ok(text: str, language: str) -> bool:
    latin = len(re.findall(r"[A-Za-z]", text))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    total = latin + cyrillic
    if total == 0:
        return False

    if language == "ru":
        return cyrillic / total >= 0.35
    if language == "en":
        return latin / total >= 0.6
    return True


def has_reasoning_markers(text: str, language: str) -> bool:
    lowered = text.lower()
    if language == "ru":
        markers = ["шаг", "сначала", "потом", "итого", "="]
    else:
        markers = ["step", "first", "then", "therefore", "="]
    return any(marker in lowered for marker in markers)


def safe_eval_expression(expression: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
        ast.Load,
    )

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported expression node: {type(node).__name__}")
    value = eval(compile(tree, "<synthetic_tool>", "eval"), {"__builtins__": {}}, {})
    return float(value)


def run_synthetic_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "get_weather":
        city = str(tool_args.get("city", "Unknown"))
        units = str(tool_args.get("units", "celsius"))
        temp_c = 7
        if units == "fahrenheit":
            temp = round((temp_c * 9 / 5) + 32, 1)
            return {"city": city, "temperature": temp, "units": "fahrenheit", "condition": "cloudy"}
        return {"city": city, "temperature": temp_c, "units": "celsius", "condition": "cloudy"}

    if tool_name == "calculator":
        expression = str(tool_args.get("expression", ""))
        result = safe_eval_expression(expression)
        if result.is_integer():
            result = int(result)
        return {"expression": expression, "result": result}

    return {"error": f"unknown tool: {tool_name}"}


def evaluate_reasoning_case(case: dict[str, Any], response_text: str) -> dict[str, Any]:
    expected_answer_ok = bool(re.search(case["expected_answer_regex"], response_text))
    lang_ok = language_ok(response_text, case["language"])
    length_ok = len(response_text) >= 35
    markers_ok = has_reasoning_markers(response_text, case["language"])

    passed = expected_answer_ok and lang_ok and length_ok

    return {
        "passed": passed,
        "checks": {
            "expected_answer_ok": expected_answer_ok,
            "language_ok": lang_ok,
            "length_ok": length_ok,
            "has_reasoning_markers": markers_ok,
        },
    }


def evaluate_tool_case(
    case: dict[str, Any],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tools = case["tools"]
    first_raw, first_generation_info = generate(
        model=model,
        tokenizer=tokenizer,
        messages=case["messages"],
        tools=tools,
        max_new_tokens=args.max_new_tokens,
        context_window=args.context_window,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    first_clean = clean_text(first_raw)

    tool_name, tool_args, parse_error = parse_tool_call(first_raw)
    tool_name_ok = tool_name == case["expected_tool_name"]
    required_args_ok = all(tool_args.get(k) is not None for k in case["required_tool_args"])
    call_ok = parse_error is None and tool_name_ok and required_args_ok

    result: dict[str, Any] = {
        "first_response_raw": first_raw,
        "first_response_clean": first_clean,
        "first_generation": first_generation_info,
        "tool_call": {
            "name": tool_name,
            "arguments": tool_args,
            "parse_error": parse_error,
            "tool_name_ok": tool_name_ok,
            "required_args_ok": required_args_ok,
        },
    }

    if not call_ok:
        result["passed"] = False
        result["checks"] = {
            "tool_call_ok": False,
            "final_answer_ok": False,
            "final_language_ok": False,
        }
        return result

    tool_result = run_synthetic_tool(tool_name, tool_args)
    assistant_tool_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                },
            }
        ],
    }
    tool_message = {
        "role": "tool",
        "content": json.dumps(tool_result, ensure_ascii=False),
    }
    followup_messages = [*case["messages"], assistant_tool_message, tool_message]

    second_raw, second_generation_info = generate(
        model=model,
        tokenizer=tokenizer,
        messages=followup_messages,
        tools=tools,
        max_new_tokens=args.max_new_tokens,
        context_window=args.context_window,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    second_clean = clean_text(second_raw)
    final_answer_ok = bool(re.search(case["expected_final_answer_regex"], second_clean))
    final_language_ok = language_ok(second_clean, case["language"])
    final_length_ok = len(second_clean) >= 12

    result["tool_result"] = tool_result
    result["second_response_raw"] = second_raw
    result["second_response_clean"] = second_clean
    result["second_generation"] = second_generation_info
    result["checks"] = {
        "tool_call_ok": True,
        "final_answer_ok": final_answer_ok,
        "final_language_ok": final_language_ok,
        "final_length_ok": final_length_ok,
    }
    result["passed"] = final_answer_ok and final_language_ok and final_length_ok
    return result


def run_eval(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started_at = time.time()
    results: list[dict[str, Any]] = []

    for case in SYNTHETIC_CASES:
        case_result: dict[str, Any] = {
            "id": case["id"],
            "type": case["type"],
            "language": case["language"],
            "input_messages": case["messages"],
        }
        if "tools" in case:
            case_result["tools"] = case["tools"]

        if case["type"] == "reasoning":
            raw, generation_info = generate(
                model=model,
                tokenizer=tokenizer,
                messages=case["messages"],
                tools=None,
                max_new_tokens=args.max_new_tokens,
                context_window=args.context_window,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            clean = clean_text(raw)
            eval_out = evaluate_reasoning_case(case, clean)
            case_result.update(
                {
                    "expected_answer_regex": case["expected_answer_regex"],
                    "response_raw": raw,
                    "response_clean": clean,
                    "generation": generation_info,
                    "checks": eval_out["checks"],
                    "passed": eval_out["passed"],
                }
            )
        else:
            case_result["expected_tool_name"] = case["expected_tool_name"]
            case_result["required_tool_args"] = case["required_tool_args"]
            case_result["expected_final_answer_regex"] = case["expected_final_answer_regex"]
            eval_out = evaluate_tool_case(case, model, tokenizer, args)
            case_result.update(eval_out)

        results.append(case_result)
        status = "PASS" if case_result["passed"] else "FAIL"
        print(f"[{status}] {case_result['id']}")

    pass_count = sum(1 for item in results if item["passed"])
    total = len(results)
    summary = {
        "pass_count": pass_count,
        "total": total,
        "pass_rate": pass_count / total if total else 0.0,
        "elapsed_sec": round(time.time() - started_at, 2),
    }

    return {"summary": summary, "cases": results}


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\n" + "=" * 80)
    print("Synthetic adapter evaluation")
    print("=" * 80)
    print(
        f"Passed {summary['pass_count']} / {summary['total']} "
        f"({summary['pass_rate'] * 100:.1f}%), elapsed {summary['elapsed_sec']}s"
    )
    print("-" * 80)
    for item in report["cases"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status:4s} | {item['type']:9s} | {item['language']:2s} | {item['id']}")


def _safe_code_block(content: str, language: str = "text") -> str:
    sanitized = (content or "").replace("```", "'''").strip()
    if not sanitized:
        sanitized = "<empty>"
    return f"```{language}\n{sanitized}\n```"


def _json_code_block(value: Any) -> str:
    return _safe_code_block(json.dumps(value, ensure_ascii=False, indent=2), language="json")


def _extract_last_user_question(messages: list[dict[str, Any]]) -> str:
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_messages:
        return ""
    return str(user_messages[-1])


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    meta = report.get("meta", {})

    lines: list[str] = []
    lines.append("# Synthetic adapter evaluation report")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Passed {summary['pass_count']} / {summary['total']} "
        f"({summary['pass_rate'] * 100:.1f}%)"
    )
    lines.append(f"- Elapsed seconds: {summary['elapsed_sec']}")
    lines.append("")
    lines.append("## Run config")
    lines.append(f"- Adapter path: `{meta.get('adapter_path', '')}`")
    lines.append(f"- Adapter weights: `{meta.get('adapter_weights', '')}`")
    lines.append(f"- Base model: `{meta.get('base_model', '')}`")
    lines.append(f"- Device: `{meta.get('device', '')}`")
    generation = meta.get("generation", {})
    lines.append(f"- Context window: `{generation.get('context_window', '')}`")
    lines.append(f"- Max new tokens: `{generation.get('max_new_tokens', '')}`")
    lines.append(f"- Temperature: `{generation.get('temperature', '')}`")
    lines.append(f"- Top p: `{generation.get('top_p', '')}`")
    lines.append(f"- Repetition penalty: `{generation.get('repetition_penalty', '')}`")
    lines.append(f"- No repeat ngram size: `{generation.get('no_repeat_ngram_size', '')}`")
    lines.append("")

    for idx, item in enumerate(report["cases"], start=1):
        lines.append("---")
        lines.append("")
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"## Case {idx} - {item['id']} - {status}")
        lines.append(f"- Type: `{item['type']}`")
        lines.append(f"- Language: `{item['language']}`")
        lines.append("")

        question = _extract_last_user_question(item.get("input_messages", []))
        lines.append("### Question")
        lines.append(_safe_code_block(question))
        lines.append("")

        lines.append("### Input")
        lines.append("#### Messages")
        lines.append(_json_code_block(item.get("input_messages", [])))
        lines.append("")
        if "tools" in item:
            lines.append("#### Tools")
            lines.append(_json_code_block(item.get("tools", [])))
            lines.append("")

        lines.append("### Output")
        if item["type"] == "reasoning":
            lines.append("#### Model response raw")
            lines.append(_safe_code_block(item.get("response_raw", "")))
            lines.append("")
            lines.append("#### Model response clean")
            lines.append(_safe_code_block(item.get("response_clean", "")))
            lines.append("")
            lines.append("#### Generation stats")
            lines.append(_json_code_block(item.get("generation", {})))
            lines.append("")
        else:
            lines.append("#### First model response raw")
            lines.append(_safe_code_block(item.get("first_response_raw", "")))
            lines.append("")
            lines.append("#### First model response clean")
            lines.append(_safe_code_block(item.get("first_response_clean", "")))
            lines.append("")
            lines.append("#### First generation stats")
            lines.append(_json_code_block(item.get("first_generation", {})))
            lines.append("")
            lines.append("#### Parsed tool call")
            lines.append(_json_code_block(item.get("tool_call", {})))
            lines.append("")
            lines.append("#### Synthetic tool result")
            lines.append(_json_code_block(item.get("tool_result", {})))
            lines.append("")
            lines.append("#### Second model response raw")
            lines.append(_safe_code_block(item.get("second_response_raw", "")))
            lines.append("")
            lines.append("#### Second model response clean")
            lines.append(_safe_code_block(item.get("second_response_clean", "")))
            lines.append("")
            lines.append("#### Second generation stats")
            lines.append(_json_code_block(item.get("second_generation", {})))
            lines.append("")

        lines.append("### Checks")
        lines.append(_json_code_block(item.get("checks", {})))
        lines.append("")

        lines.append("### Expectations")
        if item["type"] == "reasoning":
            lines.append(f"- Expected answer regex: `{item.get('expected_answer_regex', '')}`")
        else:
            lines.append(f"- Expected tool name: `{item.get('expected_tool_name', '')}`")
            lines.append(f"- Required tool args: `{item.get('required_tool_args', [])}`")
            lines.append(f"- Expected final answer regex: `{item.get('expected_final_answer_regex', '')}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    adapter_path = args.adapter_path.resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter path does not exist: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json not found in: {adapter_path}")

    adapter_weight_file = detect_adapter_weight_file(adapter_path)
    if adapter_weight_file is None:
        raise FileNotFoundError(
            "No adapter weights found. Expected adapter_model.safetensors or adapter_model.bin "
            f"in {adapter_path}"
        )

    print(f"Adapter path: {adapter_path}")
    print(f"Adapter weights: {adapter_weight_file.name}")

    model, tokenizer, base_model_name, tokenizer_patch = build_model(
        adapter_path=adapter_path,
        base_model_override=args.base_model,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"Base model: {base_model_name}")
    print(
        f"Tokenizer patch - added tokens: {len(tokenizer_patch['added_tokens'])}, "
        f"template overridden: {tokenizer_patch['template_overridden']}"
    )

    report = run_eval(model, tokenizer, args)
    report["meta"] = {
        "adapter_path": str(adapter_path),
        "adapter_weights": adapter_weight_file.name,
        "base_model": base_model_name,
        "device": str(get_model_device(model)),
        "cuda_available": torch.cuda.is_available(),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "context_window": args.context_window,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "tokenizer_patch": tokenizer_patch,
    }

    output_path = args.output_path or (adapter_path / "synthetic_eval_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown_report(report), encoding="utf-8")

    if args.json_output_path is not None:
        args.json_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print_summary(report)
    print(f"\nSaved Markdown report to: {output_path}")
    if args.json_output_path is not None:
        print(f"Saved JSON report to: {args.json_output_path}")


if __name__ == "__main__":
    main()

