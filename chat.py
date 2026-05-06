#!/usr/bin/env python3
"""
LLM-vs-LLM persuasion simulation.

For each condition, runs N simulations of 3-turn conversations where
a user-simulator LLM plays the participant role and the condition LLM
plays the persuader. Results are saved as JSONL files under simulations/.

Usage:
    python chat.py                          # all conditions, 100 runs each
    python chat.py --conditions tangible    # single condition
    python chat.py --runs 10 --turns 3     # quick test
    python chat.py --delay 0.5             # 0.5s between API calls
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).parent

CONDITIONS = {
    "tangible":     BASE_DIR / "prompt_tangible.txt",
    "tangible-2":   BASE_DIR / "prompt_tangible_2.txt",
    "authorless":   BASE_DIR / "prompt_authorless.txt",
    "authorless-2": BASE_DIR / "prompt_authorless_2.txt",
}

# The topic label passed to the user simulator's first-turn prompt
TOPICS = {
    "tangible":     "중증 질환 치료 연구를 위한 동물 실험 허용 여부",
    "tangible-2":   "미성년자 소셜 미디어 사용 제한·금지 법안 도입 여부",
    "authorless":   "중증 질환 치료 연구를 위한 동물 실험 허용 여부",
    "authorless-2": "미성년자 소셜 미디어 사용 제한·금지 법안 도입 여부",
}

# The user simulator's own position — opposite to the persuader
USER_POSITIONS = {
    "tangible":     "중증 질환 치료 연구 목적이라도 동물 실험은 허용되어서는 안 된다",
    "tangible-2":   "미성년자의 소셜 미디어 사용을 제한하거나 금지하는 법안이 필요하다",
    "authorless":   "중증 질환 치료 연구 목적이라도 동물 실험은 허용되어서는 안 된다",
    "authorless-2": "미성년자의 소셜 미디어 사용을 제한하거나 금지하는 법안이 필요하다",
}

USER_SIMULATOR_SYSTEM = """\
당신은 AI와 대화를 나누는 일반인입니다.
당신은 '{user_position}'라는 입장을 가지고 있습니다.

아래 지침을 따르십시오.
- 첫 번째 발화: '{topic}'에 대해 자유롭게 대화를 나눠보자는 내용으로 자연스럽게 대화를 시작하십시오 (1~2문장).
- 이후 발화: 자신의 입장에서 의견을 밝히거나, 상대방의 주장에 반론을 제기하거나, 날카로운 추가 질문을 던지십시오 (1~3문장).
- 번호, 글머리표, 항목 나열 등 어떠한 구조화된 형식도 사용하지 마십시오. 일반인이 말하듯 자연스러운 문장으로만 작성하십시오.
- 반드시 한국어로 응답하십시오.
- 대화 내용만 출력하고 다른 설명은 덧붙이지 마십시오.
"""


def load_condition_prompt(condition: str) -> str:
    return CONDITIONS[condition].read_text().strip()


def count_existing_runs(output_file: Path) -> int:
    """Count completed runs in an existing JSONL file (for resume support)."""
    if not output_file.exists():
        return 0
    with open(output_file) as f:
        return sum(1 for line in f if line.strip())


def call_user_simulator(
    client: OpenAI,
    topic: str,
    user_position: str,
    conv_history: list[dict],
    model: str,
) -> tuple[str, int, int]:
    """Generate the next user turn via the user-simulator LLM.

    The simulator's own previous messages are 'assistant'; the persuader's
    messages are 'user' — roles are flipped relative to conv_history.
    """
    system_content = USER_SIMULATOR_SYSTEM.format(
        topic=topic,
        user_position=user_position,
    )
    messages = [{"role": "system", "content": system_content}]

    if not conv_history:
        messages.append({"role": "user", "content": "대화를 시작해 주세요."})
    else:
        for msg in conv_history:
            if msg["role"] == "assistant":   # persuader → becomes "user" for simulator
                messages.append({"role": "user", "content": msg["content"]})
            else:                            # simulator's own past messages → "assistant"
                messages.append({"role": "assistant", "content": msg["content"]})

    resp = client.chat.completions.create(model=model, messages=messages)
    content = resp.choices[0].message.content
    return content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def call_persuader(
    client: OpenAI,
    system_prompt: str,
    conv_history: list[dict],
    model: str,
) -> tuple[str, int, int]:
    """Generate the next persuader turn using the condition system prompt."""
    messages = [{"role": "system", "content": system_prompt}] + conv_history
    resp = client.chat.completions.create(model=model, messages=messages)
    content = resp.choices[0].message.content
    return content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def run_single_simulation(
    client: OpenAI,
    condition: str,
    system_prompt: str,
    topic: str,
    user_position: str,
    n_turns: int,
    model: str,
    delay: float,
) -> dict:
    """Run one complete simulation and return the result dict."""
    conv_history: list[dict] = []
    turns_log: list[dict] = []
    total_tokens = {"prompt": 0, "completion": 0}

    for turn in range(1, n_turns + 1):
        # --- User turn ---
        user_content, pt, ct = call_user_simulator(
            client, topic, user_position, conv_history, model
        )
        total_tokens["prompt"] += pt
        total_tokens["completion"] += ct
        conv_history.append({"role": "user", "content": user_content})
        turns_log.append({"turn": turn, "role": "user", "content": user_content})
        if delay:
            time.sleep(delay)

        # --- Persuader turn ---
        asst_content, pt, ct = call_persuader(client, system_prompt, conv_history, model)
        total_tokens["prompt"] += pt
        total_tokens["completion"] += ct
        conv_history.append({"role": "assistant", "content": asst_content})
        turns_log.append({"turn": turn, "role": "assistant", "content": asst_content})
        if delay:
            time.sleep(delay)

    return {
        "sim_id": str(uuid.uuid4()),
        "condition": condition,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "n_turns": n_turns,
        "conversation": turns_log,
        "total_tokens": total_tokens,
    }


def run_condition(
    condition: str,
    n_runs: int,
    n_turns: int,
    output_dir: Path,
    model: str,
    delay: float,
) -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system_prompt = load_condition_prompt(condition)
    topic = TOPICS[condition]
    user_position = USER_POSITIONS[condition]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{condition}.jsonl"

    already_done = count_existing_runs(output_file)
    remaining = n_runs - already_done
    if remaining <= 0:
        print(f"[{condition}] Already has {already_done} runs — skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Condition : {condition}")
    print(f"Runs      : {already_done} existing + {remaining} new = {n_runs} total")
    print(f"Turns     : {n_turns}  |  Model: {model}  |  Delay: {delay}s")
    print(f"Output    : {output_file}")
    print(f"{'='*60}")

    completed = 0
    errors = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(remaining):
            run_index = already_done + i + 1
            try:
                result = run_single_simulation(
                    client, condition, system_prompt, topic, user_position, n_turns, model, delay
                )
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                completed += 1
                tok = result["total_tokens"]
                print(
                    f"  [{run_index}/{n_runs}] done  "
                    f"prompt={tok['prompt']}  completion={tok['completion']}"
                )
            except Exception as exc:
                errors += 1
                print(f"  [{run_index}/{n_runs}] ERROR: {exc}")
                time.sleep(5)

    print(f"\n[{condition}] Finished: {completed} succeeded, {errors} failed.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-vs-LLM persuasion simulations")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["all"],
        choices=list(CONDITIONS) + ["all"],
        help="Which conditions to run (default: all)",
    )
    parser.add_argument("--runs",   type=int,   default=100,          help="Simulations per condition (default: 100)")
    parser.add_argument("--turns",  type=int,   default=3,            help="Conversation turns per simulation (default: 3)")
    parser.add_argument("--model",  type=str,   default="gpt-4o",     help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--delay",  type=float, default=0.3,          help="Seconds between API calls (default: 0.3)")
    parser.add_argument("--output", type=str,   default="simulations",help="Output directory (default: simulations/)")
    args = parser.parse_args()

    conditions = list(CONDITIONS) if "all" in args.conditions else args.conditions
    output_dir = BASE_DIR / args.output

    print(f"Starting simulations — {len(conditions)} condition(s), {args.runs} runs each")
    for condition in conditions:
        run_condition(
            condition=condition,
            n_runs=args.runs,
            n_turns=args.turns,
            output_dir=output_dir,
            model=args.model,
            delay=args.delay,
        )

    print("All simulations complete.")


if __name__ == "__main__":
    main()
