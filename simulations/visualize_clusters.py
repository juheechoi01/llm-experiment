#!/usr/bin/env python3
"""
UMAP + Plotly 클러스터 시각화
  - 모든 point statement 임베딩 → UMAP 2D → K-means 색상 + 조건별 마커
  - 결과: simulations/cluster_visualization.html
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")
client = OpenAI()


# ─── 데이터 로딩 & 파싱 (check_operationalization.py와 동일 로직) ──────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_assistant_turns(sims):
    turns = []
    for sim in sims:
        for t in sim["conversation"]:
            if t["role"] == "assistant":
                turns.append({
                    "sim_id": sim["sim_id"],
                    "condition": sim["condition"],
                    "turn": t["turn"],
                    "content": t["content"],
                })
    return turns


def parse_response(content):
    content = content.strip()
    points = {}
    numbered = re.findall(r'(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\Z)', content, re.DOTALL)
    if numbered:
        for num_str, text in numbered:
            num = int(num_str)
            if 1 <= num <= 3:
                text = text.strip()
                newline_split = re.split(r'\n\s+', text, maxsplit=1)
                if len(newline_split) == 2:
                    stmt = newline_split[0].strip().rstrip('.')
                    sub = newline_split[1].strip()
                else:
                    parts = re.split(r'\.\s+(?=[가-힣A-Z\d])', text, maxsplit=1)
                    stmt = parts[0].strip() if len(parts) == 2 else text.strip().rstrip('.')
                    sub = parts[1].strip() if len(parts) == 2 else ""
                points[num] = {"statement": stmt, "subclaim": sub, "full": text}
    if not points:
        bullets = re.findall(r'^[-•]\s+(.+?)(?=\n[-•]|\Z)', content, re.DOTALL | re.MULTILINE)
        for i, text in enumerate(bullets[:3], 1):
            text = text.strip()
            stmt = text.split('\n')[0].strip().rstrip('.')
            points[i] = {"statement": stmt, "subclaim": "", "full": text}
    return points


# ─── 임베딩 ────────────────────────────────────────────────────────────────────

def get_embeddings(texts, batch_size=500, model="text-embedding-3-small"):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(input=texts[i:i + batch_size], model=model)
        all_embs.extend([e.embedding for e in resp.data])
    return np.array(all_embs, dtype=np.float32)


# ─── GPT 클러스터 레이블링 ─────────────────────────────────────────────────────

def label_clusters(cluster_texts: dict, k: int) -> dict:
    prompt_lines = []
    for cid in range(k):
        examples = cluster_texts[cid][:8]
        ex_str = "\n".join(f"  - {e}" for e in examples)
        prompt_lines.append(f"Cluster {cid}:\n{ex_str}")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "아래는 동물실험 찬성 논증 문장들을 K-means로 클러스터링한 결과입니다.\n"
                "각 클러스터의 핵심 논증 주제를 한국어로 10단어 이내로 이름 붙여주세요.\n"
                '응답 형식 (JSON): {"labels": {"0": "주제명", "1": "주제명", ...}}\n\n'
                + "\n\n".join(prompt_lines)
            )
        }],
        response_format={"type": "json_object"},
    )
    raw = json.loads(resp.choices[0].message.content).get("labels", {})
    return {int(k): v for k, v in raw.items()}


# ─── 메인 ──────────────────────────────────────────────────────────────────────

def main(k: int = 6):
    print("데이터 로딩...")
    authorless = load_jsonl(DATA_DIR / "authorless.jsonl")
    tangible   = load_jsonl(DATA_DIR / "tangible.jsonl")
    all_turns  = get_assistant_turns(authorless + tangible)

    # Point statements 수집
    records = []
    for turn in all_turns:
        for pt_num, pt in parse_response(turn["content"]).items():
            if pt["statement"].strip():
                records.append({
                    "text":      pt["statement"],
                    "subclaim":  pt["subclaim"],
                    "condition": turn["condition"],
                    "turn":      turn["turn"],
                    "point":     pt_num,
                    "sim_id":    turn["sim_id"],
                })

    texts = [r["text"] for r in records]
    print(f"  총 {len(texts)}개 statement 임베딩 계산 중...")
    embs = get_embeddings(texts)
    embs_norm = normalize(embs)
    

    # K-means
    print(f"  K-means (k={k}) 클러스터링...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(embs_norm)
    for i, rec in enumerate(records):
        rec["cluster"] = int(cluster_ids[i])

    # GPT 레이블
    print("  클러스터 레이블링...")
    cluster_texts = defaultdict(list)
    for rec in records:
        cluster_texts[rec["cluster"]].append(rec["text"])
    labels = label_clusters(cluster_texts, k)
    print("  레이블:", labels)

    # UMAP
    print("  UMAP 차원 축소 중 (2D)...")
    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15)
    coords = reducer.fit_transform(embs_norm)
    for i, rec in enumerate(records):
        rec["x"] = float(coords[i, 0])
        rec["y"] = float(coords[i, 1])

    # ─── Plotly 시각화 ────────────────────────────────────────────────────────

    CONDITION_SYMBOL = {"tangible": "circle", "authorless": "diamond"}
    CONDITION_LABEL  = {"tangible": "Tangible (기관 인용)", "authorless": "Authorless (익명)"}

    # 팔레트: 클러스터 수만큼
    PALETTE = [
        "#E15759", "#4E79A7", "#F28E2B", "#76B7B2",
        "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    ]

    fig = go.Figure()

    for cid in range(k):
        cluster_label = labels.get(cid, f"Cluster {cid}")
        for cond in ["tangible", "authorless"]:
            subset = [r for r in records if r["cluster"] == cid and r["condition"] == cond]
            if not subset:
                continue

            hover_texts = [
                f"<b>Cluster {cid}: {cluster_label}</b><br>"
                f"조건: {cond}  |  Turn {r['turn']}  |  Point {r['point']}<br><br>"
                f"<b>Statement:</b> {r['text']}<br><br>"
                f"<b>Sub-claim:</b> {r['subclaim'][:120]}{'…' if len(r['subclaim']) > 120 else ''}"
                for r in subset
            ]

            fig.add_trace(go.Scatter(
                x=[r["x"] for r in subset],
                y=[r["y"] for r in subset],
                mode="markers",
                name=f"C{cid} {cluster_label[:12]} / {cond}",
                legendgroup=f"cluster_{cid}",
                marker=dict(
                    color=PALETTE[cid % len(PALETTE)],
                    symbol=CONDITION_SYMBOL[cond],
                    size=7,
                    opacity=0.75,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            ))

    # 클러스터 중심 레이블 (텍스트 annotation)
    for cid in range(k):
        subset = [r for r in records if r["cluster"] == cid]
        cx = float(np.mean([r["x"] for r in subset]))
        cy = float(np.mean([r["y"] for r in subset]))
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>C{cid}</b><br>{labels.get(cid, '')}",
            showarrow=False,
            font=dict(size=11, color=PALETTE[cid % len(PALETTE)]),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=PALETTE[cid % len(PALETTE)],
            borderwidth=1,
            borderpad=3,
        )

    # 범례용 dummy trace (조건 구분 마커)
    for cond, sym in CONDITION_SYMBOL.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=CONDITION_LABEL[cond],
            legendgroup=f"cond_{cond}",
            marker=dict(symbol=sym, size=10, color="gray"),
            showlegend=True,
        ))

    t_n = sum(1 for r in records if r["condition"] == "tangible")
    a_n = sum(1 for r in records if r["condition"] == "authorless")

    fig.update_layout(
        title=dict(
            text=(
                f"Point Statement 클러스터 시각화 (UMAP 2D, K-means k={k})<br>"
                f"<sup>tangible {t_n}개 (●)  /  authorless {a_n}개 (◆)  —  색상 = 클러스터</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
        plot_bgcolor="#f9f9f9",
        legend=dict(
            title="클러스터 / 조건",
            itemsizing="constant",
            font=dict(size=11),
        ),
        width=1100,
        height=750,
        hovermode="closest",
    )

    out_path = DATA_DIR / "cluster_visualization.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\n시각화 저장: {out_path}")
    print("브라우저에서 열어서 확인하세요.")


if __name__ == "__main__":
    main(k=6)
