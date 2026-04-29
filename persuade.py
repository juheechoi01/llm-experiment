#!/usr/bin/env python3
"""Persuasion chat: LLM argues against the user's stated attitude."""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROMPT_FILES = {
    "1": ("authorless", os.path.join(PROMPT_DIR, "persuade_authorless.txt")),
    "2": ("tangible",   os.path.join(PROMPT_DIR, "persuade_tangible.txt")),
}

def load_template(path: str) -> str:
    with open(path) as f:
        return f.read().strip()

def get_response(client: OpenAI, history: list) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=history,
    )
    usage = response.usage
    msg = response.choices[0].message.content
    print(f"\nGPT-4o: {msg}")
    print(f"\n[tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}]\n")
    return msg

def collect_inputs() -> tuple[str, int, str]:
    print("\n" + "="*60)
    print("설득 실험 채팅")
    print("="*60)

    print("\n[프롬프트 조건 선택]")
    print("  1. Authorless — 출처 없이 일반적 근거로 설득")
    print("  2. Tangible   — 특정 기관/연구자를 명시하며 설득")
    while True:
        choice = input("선택 (1 or 2): ").strip()
        if choice in PROMPT_FILES:
            condition_name, prompt_path = PROMPT_FILES[choice]
            break
        print("1 또는 2를 입력해주세요.")

    attitude = input("\n설득할 태도를 입력하세요: ").strip()
    while not attitude:
        attitude = input("태도를 입력해주세요: ").strip()

    while True:
        raw = input("그 태도에 동의하는 정도를 입력하세요 (0~100): ").strip()
        if raw.isdigit() and 0 <= int(raw) <= 100:
            score = int(raw)
            break
        print("0에서 100 사이의 숫자를 입력해주세요.")

    return attitude, score, prompt_path, condition_name

def chat():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    attitude, score, prompt_path, condition_name = collect_inputs()

    template = load_template(prompt_path)
    system_prompt = template.format(attitude=attitude, score=score)

    history = [{"role": "system", "content": system_prompt}]

    print(f"\n[조건: {condition_name} | 태도: '{attitude}' | 동의 점수: {score}/100]")
    print("/quit - 종료  |  /reset - 대화 초기화  |  /system - 프롬프트 보기")
    print("="*60 + "\n")

    # LLM speaks first
    try:
        opening = get_response(client, history)
        history.append({"role": "assistant", "content": opening})
    except Exception as e:
        print(f"[Error] {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("종료합니다.")
            break
        elif user_input == "/system":
            print(f"\n[System prompt]\n{history[0]['content']}\n")
            continue
        elif user_input == "/reset":
            history = [{"role": "system", "content": system_prompt}]
            print("[대화 초기화됨]\n")
            try:
                opening = get_response(client, history)
                history.append({"role": "assistant", "content": opening})
            except Exception as e:
                print(f"[Error] {e}\n")
            continue

        history.append({"role": "user", "content": user_input})
        try:
            reply = get_response(client, history)
            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"\n[Error] {e}\n")
            history.pop()

if __name__ == "__main__":
    chat()
