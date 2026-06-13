"""Open-source-safe prompt generation and local Ollama inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from .constants import AOA_BINS_ZH, COUNTRY_LABEL_ZH, EYE_TARGETS, HSK_BINS_ZH, LOE_BINS_ZH, TARGET_LABEL_ZH
from .io_utils import format_token_value_pairs, read_table, safe_literal_list, write_table


def map_stage(value: Any, mapping: Dict[str, str], unknown: str = "未知阶段") -> str:
    try:
        val = float(value)
    except Exception:
        return unknown
    for interval, name in mapping.items():
        low, high = interval.strip("()[]").split(",")
        low, high = float(low), float(high)
        if val > low and val <= high:
            return name
    return unknown


def call_ollama_chat(
    prompt: str,
    model_name: str = "llama3.3:70b",
    endpoint: str = "http://localhost:11434/v1/chat/completions",
    temperature: float = 0.7,
    timeout: int = 300,
) -> str:
    data = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    resp = requests.post(endpoint, json=data, headers={"Content-Type": "application/json"}, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    return content.strip()


def build_group_prompt(row: pd.Series, target: str, n_shot: int, group_type: str) -> str:
    cname = TARGET_LABEL_ZH[target]
    tokens = safe_literal_list(row["IA_LABEL"])
    sent = " ".join(map(str, tokens))
    if group_type == "country":
        group_name = COUNTRY_LABEL_ZH.get(str(row.get("country", ""))[:1], "未知国家")
        subject = f"{group_name}汉语学习者群体"
    elif group_type == "AoA":
        subject = f"{row.get('AoA_bin', '未知起始阶段')}起始学习汉语的二语学习者群体"
    elif group_type == "LoE":
        subject = f"处于{row.get('LoE_bin', '未知学习阶段')}的二语学习者群体"
    elif group_type == "HSK":
        subject = f"{row.get('HSK_bin', '未知水平阶段')}汉语水平的二语学习者群体"
    else:
        subject = "二语学习者群体"

    if n_shot <= 0:
        fmt = format_token_value_pairs(tokens)
        return f"""对{subject}：
请直接输出该群体对整句话中所有分词的{cname}（{target}），一次性给出，不要解释。
汉语文本：{sent}
请严格用 Python 列表 [] 表示，列表中每个元素是字符串，形式为“词-数值”，不要写成字典，不要出现冒号。
示例格式（仅示范，不是答案）：{fmt}
请给出{subject}{cname}："""

    examples: List[str] = []
    for k in range(1, n_shot + 1):
        ex_tokens = safe_literal_list(row.get(f"example{k}_IA_LABEL", row["IA_LABEL"]))
        ex_vals = safe_literal_list(row.get(f"example{k}_{target}", []))
        examples.append(f"示例{k}{cname}：{format_token_value_pairs(ex_tokens, ex_vals)}")
    examples_text = "\n".join(examples)
    return f"""对{subject}：
请直接输出该群体对整句话中所有分词的{cname}（{target}），一次性给出，不要解释。
以下是{n_shot}个示例（仅示范，不是答案），目标文本为：{sent}
{examples_text}
请给出{subject}{cname}："""


def build_individual_prompt(row: pd.Series, target: str, n_shot: int) -> str:
    cname = TARGET_LABEL_ZH[target]
    tokens = safe_literal_list(row["IA_LABEL"])
    sent = " ".join(map(str, tokens))
    country = COUNTRY_LABEL_ZH.get(str(row.get("DATA_FILE", ""))[:1], "未知背景")
    aoa_stage = map_stage(row.get("AoA"), AOA_BINS_ZH)
    loe_stage = map_stage(row.get("LoE"), LOE_BINS_ZH)
    hsk_stage = map_stage(row.get("HSK"), HSK_BINS_ZH)
    target_bg = (
        f"习得年龄：{row.get('AoA')}（{aoa_stage}），"
        f"汉语学习年限：{row.get('LoE')}（{loe_stage}），"
        f"汉语水平阶段：{row.get('HSK')}（{hsk_stage}），母语背景：{country}"
    )
    if n_shot <= 0:
        fmt = format_token_value_pairs(tokens)
        return f"""请根据以下汉语学习者的个体背景信息，预测其在阅读过程中对分词的{cname}（{target}）。
目标学习者背景：{target_bg}
目标文本：{sent}
请严格使用 Python 列表 [] 表示（可直接被 ast.literal_eval() 解析），元素形式为“词-数值”，不要解释，不要输出其他内容。
示例格式（仅示范，不是答案）：{fmt}
{cname}="""

    examples: List[str] = []
    for k in range(1, n_shot + 1):
        ex_tokens = safe_literal_list(row.get(f"example{k}_IA_LABEL", []))
        ex_vals = safe_literal_list(row.get(f"example{k}_{target}", []))
        examples.append(f"第{k}个学习者：{cname} = {format_token_value_pairs(ex_tokens, ex_vals)}")
    return f"""以下是{n_shot}个学习者示例（仅示范，不是答案）：
{chr(10).join(examples)}
请根据以上示例与下方学习者的个体背景信息，预测该学习者在阅读过程中的{cname}（{target}）分布。
目标学习者背景：{target_bg}
目标文本：{sent}
请严格使用 Python 列表 [] 表示输出（可直接被 ast.literal_eval() 解析），元素形式为“词-数值”，不要解释，不要输出其他内容。
{cname}="""


def run_prompt_table(
    input_file: str | Path,
    output_dir: str | Path,
    mode: str,
    group_type: str = "LoE",
    targets: List[str] | None = None,
    max_shot: int = 5,
    model_name: str = "llama3.3:70b",
    dry_run: bool = False,
) -> None:
    df = read_table(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = targets or EYE_TARGETS
    for target in targets:
        for n_shot in range(max_shot + 1):
            out = df.copy()
            prompts, outputs = [], []
            for _, row in tqdm(out.iterrows(), total=len(out), desc=f"{mode}:{target}:shot{n_shot}"):
                prompt = build_group_prompt(row, target, n_shot, group_type) if mode == "group" else build_individual_prompt(row, target, n_shot)
                prompts.append(prompt)
                outputs.append(prompt if dry_run else call_ollama_chat(prompt, model_name=model_name))
            out["PROMPT"] = prompts
            out["OUTPUT"] = outputs
            write_table(out, output_dir / f"output_{mode}_{group_type}_{target}_shot{n_shot}.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM prompts or run local Ollama inference.")
    parser.add_argument("mode", choices=["group", "individual"])
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_type", default="LoE")
    parser.add_argument("--model_name", default="llama3.3:70b")
    parser.add_argument("--max_shot", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_prompt_table(**vars(args))


if __name__ == "__main__":
    main()
