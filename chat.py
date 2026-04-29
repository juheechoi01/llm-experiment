#!/usr/bin/env python3
"""GPT-4o terminal chat for prompt engineering experiments."""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def load_system_prompt(path: str | None) -> str:
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return DEFAULT_SYSTEM_PROMPT

def chat(system_prompt: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    history = [{"role": "system", "content": system_prompt}]

    print(f"\n{'='*60}")
    print("GPT-4o Chat  |  /system - 시스템 프롬프트 보기  |  /reset - 대화 초기화  |  /quit - 종료")
    print(f"{'='*60}\n")

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
            continue
        elif user_input.startswith("/system "):
            new_prompt = user_input[len("/system "):].strip()
            system_prompt = new_prompt
            history = [{"role": "system", "content": system_prompt}]
            print(f"[시스템 프롬프트 변경 및 대화 초기화됨]\n")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=history,
            )
            assistant_msg = response.choices[0].message.content
            history.append({"role": "assistant", "content": assistant_msg})

            usage = response.usage
            print(f"\nGPT-4o: {assistant_msg}")
            print(f"\n[tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}]\n")

        except Exception as e:
            print(f"\n[Error] {e}\n")
            history.pop()

if __name__ == "__main__":
    system_prompt_file = sys.argv[1] if len(sys.argv) > 1 else None
    system_prompt = load_system_prompt(system_prompt_file)
    chat(system_prompt)
