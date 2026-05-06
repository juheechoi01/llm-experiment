#!/usr/bin/env python3
"""
Operationalization Check: User 반박에 대한 LLM 응답 직접성 분석

프롬프트 규칙: "첫 번째 point는 반드시 user의 반박을 직접 반론해야 한다"
→ Turn 2, 3에서 user 반박 vs LLM Point 1을 GPT로 분류

분류:
  direct  : Point 1이 user 반박의 핵심 주장을 직접 반론
  partial : 관련 있지만 핵심을 비껴감
  evasion : user 반박을 무시하고 다른 논점으로 전환

출력: directness_results.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")
client = OpenAI()


# ─── 데이터 로딩 & 파싱 ────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_point1(content):
    """응답에서 Point 1 statement만 추출."""
    content = content.strip()
    numbered = re.findall(r'1\.\s+(.+?)(?=\n\s*2\.|\Z)', content, re.DOTALL)
    if numbered:
        text = numbered[0].strip()
        newline_split = re.split(r'\n\s+', text, maxsplit=1)
        if len(newline_split) == 2:
            return newline_split[0].strip().rstrip('.')
        parts = re.split(r'\.\s+(?=[가-힣A-Z\d])', text, maxsplit=1)
        return parts[0].strip() if len(parts) == 2 else text.strip().rstrip('.')
    bullets = re.findall(r'^[-•]\s+(.+?)(?=\n[-•]|\Z)', content, re.DOTALL | re.MULTILINE)
    if bullets:
        return bullets[0].split('\n')[0].strip().rstrip('.')
    return ""


def build_pairs(simulations):
    """Turn 2, 3에서 (user 반박, LLM Point 1) 쌍 수집."""
    pairs = []
    for sim in simulations:
        conv = sim["conversation"]
        for i, turn in enumerate(conv):
            if turn["role"] == "user" and turn["turn"] in [2, 3]:
                # 바로 다음 assistant 응답 찾기
                next_turns = [t for t in conv if t["role"] == "assistant" and t["turn"] == turn["turn"]]
                if not next_turns:
                    continue
                asst = next_turns[0]
                point1 = parse_point1(asst["content"])
                if point1:
                    pairs.append({
                        "sim_id":    sim["sim_id"],
                        "condition": sim["condition"],
                        "turn":      turn["turn"],
                        "user_objection": turn["content"].strip(),
                        "point1":    point1,
                    })
    return pairs


# ─── GPT 분류 (배치) ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 토론 분석 전문가입니다.
사용자의 반박(objection)과 LLM의 첫 번째 논점(Point 1)을 보고,
LLM이 반박에 얼마나 직접적으로 대응했는지 분류합니다.

분류 기준:
- direct : Point 1이 user 반박의 핵심 주장을 명시적으로 반론함
- partial: user 반박과 관련 있지만 핵심을 비껴가거나 일부만 다룸
- evasion: user 반박을 사실상 무시하고 다른 논점으로 전환함

응답 형식 (JSON):
{"results": [{"id": <int>, "classification": "direct|partial|evasion", "reason": "한 문장 이유"}]}"""


def classify_batch(pairs_batch: list[dict]) -> list[dict]:
    items = []
    for i, p in enumerate(pairs_batch):
        items.append(
            f"[{i}]\n"
            f"User 반박: {p['user_objection'][:300]}\n"
            f"LLM Point 1: {p['point1']}"
        )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "\n\n---\n\n".join(items)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content).get("results", [])


def classify_all(pairs: list[dict], batch_size: int = 15) -> list[dict]:
    results = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        print(f"  분류 중... {i+1}~{min(i+batch_size, len(pairs))} / {len(pairs)}")
        batch_results = classify_batch(batch)
        for res in batch_results:
            idx = res["id"]
            if idx < len(batch):
                results.append({**batch[idx], **res})
    return results


# ─── 통계 & 시각화 ─────────────────────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in results:
        cond = r["condition"]
        turn = r["turn"]
        cls  = r["classification"]
        stats["overall"][cond][cls]        += 1
        stats[f"turn{turn}"][cond][cls]    += 1

    # 비율 계산
    def to_pct(d):
        total = sum(d.values())
        return {k: round(100 * v / total, 1) for k, v in d.items()} if total else {}

    out = {}
    for scope, cond_dict in stats.items():
        out[scope] = {}
        for cond, cls_dict in cond_dict.items():
            total = sum(cls_dict.values())
            out[scope][cond] = {
                "counts": dict(cls_dict),
                "pct":    to_pct(cls_dict),
                "total":  total,
            }
    return out


def make_visualization(results: list[dict], stats: dict):
    LABELS  = ["direct", "partial", "evasion"]
    COLORS  = {"direct": "#59A14F", "partial": "#F28E2B", "evasion": "#E15759"}
    SCOPES  = ["overall", "turn2", "turn3"]
    SLABELS = {"overall": "전체", "turn2": "Turn 2", "turn3": "Turn 3"}
    CONDS   = ["tangible", "authorless"]

    t_direct = stats["overall"].get("tangible",   {}).get("pct", {}).get("direct", 0)
    a_direct = stats["overall"].get("authorless", {}).get("pct", {}).get("direct", 0)
    t_total  = stats["overall"].get("tangible",   {}).get("total", 0)
    a_total  = stats["overall"].get("authorless", {}).get("total", 0)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "전체 비교 (stacked %)",
            "Turn 2 비교 (stacked %)",
            "Turn 3 비교 (stacked %)",
            "Direct 비율 히트맵 (조건 × 구간)",
            "Evasion 비율 히트맵 (조건 × 구간)",
            "조건별 전체 분류 수 (건)",
        ],
        row_heights=[0.52, 0.48],
        horizontal_spacing=0.10,
        vertical_spacing=0.20,
    )

    # ── Row 1: 구간별 stacked bar ──────────────────────────────────────────────
    for col, scope in enumerate(SCOPES, start=1):
        for lbl in LABELS:
            fig.add_trace(go.Bar(
                name=lbl,
                x=CONDS,
                y=[stats[scope].get(c, {}).get("pct", {}).get(lbl, 0) for c in CONDS],
                marker_color=COLORS[lbl],
                showlegend=(col == 1),
                text=[f"{stats[scope].get(c,{}).get('pct',{}).get(lbl,0):.0f}%"
                      for c in CONDS],
                textposition="inside",
                hovertemplate=f"{SLABELS[scope]} {lbl}: %{{y:.1f}}%<extra></extra>",
            ), row=1, col=col)

    # ── Row 2, col 1-2: 히트맵 ────────────────────────────────────────────────
    for col, metric in enumerate(["direct", "evasion"], start=1):
        z = [[stats[sc].get(c, {}).get("pct", {}).get(metric, 0)
              for sc in SCOPES] for c in CONDS]
        cscale = [[0, "#FFF5F0"], [0.5, "#FC8D59"], [1, "#59A14F"]] if metric == "direct" \
            else [[0, "#EAF3FB"], [0.5, "#6BAED6"], [1, "#E15759"]]
        fig.add_trace(go.Heatmap(
            z=z,
            x=[SLABELS[s] for s in SCOPES],
            y=["Tangible", "Authorless"],
            colorscale=cscale,
            zmin=0, zmax=100,
            text=[[f"{v:.0f}%" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=14, color="black"),
            showscale=False,
            hovertemplate=f"{metric} | %{{y}} %{{x}}: %{{z:.1f}}%<extra></extra>",
        ), row=2, col=col)

    # ── Row 2, col 3: 분류 건수 grouped bar ───────────────────────────────────
    for lbl in LABELS:
        fig.add_trace(go.Bar(
            name=lbl,
            x=CONDS,
            y=[stats["overall"].get(c, {}).get("counts", {}).get(lbl, 0) for c in CONDS],
            marker_color=COLORS[lbl],
            showlegend=False,
            text=[stats["overall"].get(c, {}).get("counts", {}).get(lbl, 0) for c in CONDS],
            textposition="outside",
            hovertemplate=f"{lbl}: %{{y}}건<extra></extra>",
        ), row=2, col=3)

    fig.update_layout(
        title=dict(
            text=(
                f"User 반박 직접성 분석 — Tangible vs Authorless<br>"
                f"<sup>Turn 2·3 대상 | Tangible {t_total}개 (Direct {t_direct:.0f}%)"
                f" | Authorless {a_total}개 (Direct {a_direct:.0f}%)</sup>"
            ),
            x=0.5, font=dict(size=14),
        ),
        barmode="stack",
        plot_bgcolor="#f9f9f9",
        legend=dict(title="분류", orientation="h", y=1.06, x=0.5, xanchor="center"),
        height=720, width=1150,
        margin=dict(t=120, b=60),
    )

    for col in [1, 2, 3]:
        fig.update_yaxes(title_text="%", range=[0, 115], row=1, col=col)
    fig.update_yaxes(title_text="건수", row=2, col=3)
    fig.update_layout(barmode="stack")

    out_path = DATA_DIR / "directness_visualization.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\n시각화 저장: {out_path}")


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("데이터 로딩...")
    authorless = load_jsonl(DATA_DIR / "authorless.jsonl")
    tangible   = load_jsonl(DATA_DIR / "tangible.jsonl")

    pairs = build_pairs(authorless + tangible)
    print(f"  분석 대상 쌍: {len(pairs)}개 "
          f"(tangible {sum(1 for p in pairs if p['condition']=='tangible')}, "
          f"authorless {sum(1 for p in pairs if p['condition']=='authorless')})")

    print("\nGPT-4o-mini로 직접성 분류 중...")
    results = classify_all(pairs)

    stats = compute_stats(results)

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("직접성 분류 결과")
    print("="*60)
    print(f"\n  {'':12} {'direct':>8} {'partial':>8} {'evasion':>8}  (n)")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}  ---")
    for scope in ["overall", "turn2", "turn3"]:
        lbl = {"overall": "전체", "turn2": "Turn 2", "turn3": "Turn 3"}[scope]
        for cond in ["tangible", "authorless"]:
            d = stats[scope].get(cond, {})
            p = d.get("pct", {})
            n = d.get("total", 0)
            print(f"  {lbl+' '+cond:20}  {p.get('direct',0):>6.1f}%  {p.get('partial',0):>6.1f}%  {p.get('evasion',0):>6.1f}%  ({n})")
        print()

    # evasion 사례 샘플
    evasion_cases = [r for r in results if r["classification"] == "evasion"]
    if evasion_cases:
        print(f"  [Evasion 사례 샘플 (최대 4개)]")
        for r in evasion_cases[:4]:
            print(f"\n  조건: {r['condition']} | Turn {r['turn']}")
            print(f"  User 반박: {r['user_objection'][:120]}...")
            print(f"  Point 1:   {r['point1'][:100]}")
            print(f"  판단 이유: {r.get('reason', '')}")

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    output = {
        "stats":   stats,
        "results": [{k: v for k, v in r.items() if k != "id"} for r in results],
    }
    out_path = DATA_DIR / "directness_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")

    make_visualization(results, stats)
    print("분석 완료.")


if __name__ == "__main__":
    main()
