#!/usr/bin/env python3
"""
BERTopic 문서 UMAP 시각화 — 조건별 분리 (Authorless | Tangible)
  - 동일 BERTopic 파라미터로 재실행 (임베딩 캐시 사용, GPT 호출 없음)
  - 색상 = 토픽, 반대 조건은 회색 배경으로 표시
  - 결과: bertopic_docs_custom.html
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")

PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#D37295", "#FABFD2", "#8CD17D", "#B6992D", "#499894",
    "#86BCB6", "#A0CBE8", "#FF9D9A", "#79706E", "#D7B5A6",
    "#D4A6C8",
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_assistant_turns(sims):
    turns = []
    for sim in sims:
        for t in sim["conversation"]:
            if t["role"] == "assistant":
                turns.append({
                    "sim_id":    sim["sim_id"],
                    "condition": sim["condition"],
                    "turn":      t["turn"],
                    "content":   t["content"],
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
                sp = re.split(r'\n\s+', text, maxsplit=1)
                if len(sp) == 2:
                    stmt, sub = sp[0].strip().rstrip('.'), sp[1].strip()
                else:
                    parts = re.split(r'\.\s+(?=[가-힣A-Z\d])', text, maxsplit=1)
                    stmt = parts[0].strip() if len(parts) == 2 else text.strip().rstrip('.')
                    sub  = parts[1].strip() if len(parts) == 2 else ""
                points[num] = {"statement": stmt, "subclaim": sub}
    if not points:
        bullets = re.findall(r'^[-•]\s+(.+?)(?=\n[-•]|\Z)', content, re.DOTALL | re.MULTILINE)
        for i, text in enumerate(bullets[:3], 1):
            text = text.strip()
            points[i] = {"statement": text.split('\n')[0].strip().rstrip('.'), "subclaim": ""}
    return points


def main():
    print("데이터 로딩...")
    authorless = load_jsonl(DATA_DIR / "authorless.jsonl")
    tangible   = load_jsonl(DATA_DIR / "tangible.jsonl")
    all_turns  = get_assistant_turns(authorless + tangible)

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
                })

    texts = [r["text"] for r in records]
    print(f"  {len(texts)}개 statement 로드 완료")

    embs      = np.load(DATA_DIR / "embeddings_cache.npy")
    embs_norm = normalize(embs)

    # BERTopic 재실행 (same params, 임베딩 캐시 사용, GPT 호출 없음)
    print("BERTopic 재실행 중...")
    umap_model = UMAP(
        n_components=5, n_neighbors=15, min_dist=0.0,
        metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=25, min_samples=5,
        metric="euclidean", cluster_selection_method="eom", prediction_data=True,
    )
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer, calculate_probabilities=False, verbose=False,
    )
    topic_ids, _ = topic_model.fit_transform(texts, embeddings=embs_norm)

    n_noise  = sum(1 for t in topic_ids if t == -1)
    n_topics = len(set(t for t in topic_ids if t != -1))
    print(f"  토픽 수: {n_topics}  noise: {n_noise}")

    # 저장된 레이블 로드 (GPT 재호출 없음)
    with open(DATA_DIR / "bertopic_results.json", encoding="utf-8") as f:
        saved = json.load(f)
    saved_labels = {int(k): v.get("label", f"Topic {k}") for k, v in saved["topics"].items()}

    # 각 record에 topic_id 부여
    for i, rec in enumerate(records):
        rec["topic"] = int(topic_ids[i])

    # UMAP 2D (전체 데이터로 fit → 동일 좌표계 보장)
    print("UMAP 2D 계산 중...")
    umap_2d = UMAP(
        n_components=2, random_state=42, min_dist=0.1,
        n_neighbors=15, metric="cosine",
    )
    coords = umap_2d.fit_transform(embs_norm)
    for i, rec in enumerate(records):
        rec["x"] = float(coords[i, 0])
        rec["y"] = float(coords[i, 1])

    x_all   = [r["x"] for r in records]
    y_all   = [r["y"] for r in records]
    x_range = [min(x_all) - 0.5, max(x_all) + 0.5]
    y_range = [min(y_all) - 0.5, max(y_all) + 0.5]

    unique_topics  = sorted(set(topic_ids))
    non_noise_tids = [t for t in unique_topics if t != -1]

    # ── Figure: 1행 2열 ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Authorless", "Tangible"],
        horizontal_spacing=0.06,
    )

    for col, cond in enumerate(["authorless", "tangible"], start=1):
        subset_all = [r for r in records if r["condition"] == cond]
        other      = [r for r in records if r["condition"] != cond]

        # 반대 조건 회색 배경
        fig.add_trace(go.Scatter(
            x=[r["x"] for r in other],
            y=[r["y"] for r in other],
            mode="markers",
            marker=dict(color="#DDDDDD", size=4, opacity=0.3),
            hoverinfo="skip", showlegend=False, name="",
        ), row=1, col=col)

        # Noise
        noise_sub = [r for r in subset_all if r["topic"] == -1]
        if noise_sub:
            fig.add_trace(go.Scatter(
                x=[r["x"] for r in noise_sub],
                y=[r["y"] for r in noise_sub],
                mode="markers",
                name="Noise",
                legendgroup="noise",
                showlegend=(col == 1),
                marker=dict(color="#CCCCCC", size=4, opacity=0.45,
                            line=dict(width=0.3, color="white")),
                text=[
                    f"<b>Noise</b><br>Turn {r['turn']}  |  Point {r['point']}<br>"
                    f"<b>Statement:</b> {r['text']}"
                    for r in noise_sub
                ],
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=col)

        # 토픽별 색상
        for tid in non_noise_tids:
            label = saved_labels.get(tid, f"Topic {tid}")
            color = PALETTE[tid % len(PALETTE)]
            subset = [r for r in subset_all if r["topic"] == tid]
            if not subset:
                continue
            fig.add_trace(go.Scatter(
                x=[r["x"] for r in subset],
                y=[r["y"] for r in subset],
                mode="markers",
                name=f"T{tid}: {label[:18]}",
                legendgroup=f"topic_{tid}",
                showlegend=(col == 1),
                marker=dict(
                    color=color, size=7, opacity=0.78,
                    line=dict(width=0.5, color="white"),
                ),
                text=[
                    f"<b>T{tid}: {label}</b><br>"
                    f"Turn {r['turn']}  |  Point {r['point']}<br>"
                    f"<b>Statement:</b> {r['text']}"
                    for r in subset
                ],
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=col)

        # 조건별 토픽 분포 annotation (우하단)
        total_cond = len(subset_all)
        xref_str = f"x{col} domain" if col > 1 else "x domain"
        yref_str = f"y{col} domain" if col > 1 else "y domain"
        fig.add_annotation(
            xref=xref_str, yref=yref_str,
            x=0.99, y=0.01, row=1, col=col,
            text=f"n={total_cond}<br>noise {100*n_noise/len(records):.0f}%",
            showarrow=False,
            font=dict(size=9, color="#555"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#ccc", borderwidth=1, borderpad=4,
            align="right",
        )

    n_t = sum(1 for r in records if r["condition"] == "tangible")
    n_a = sum(1 for r in records if r["condition"] == "authorless")
    kl  = saved.get("kl_divergence", "?")

    fig.update_layout(
        title=dict(
            text=(
                f"BERTopic — 조건별 UMAP 2D 비교<br>"
                f"<sup>동일 UMAP 좌표계  |  회색 = 반대 조건(참고용)  |  "
                f"색상 = 토픽  |  {n_topics}개 토픽  |  "
                f"KL divergence = {kl}  |  T:{n_t} / A:{n_a}</sup>"
            ),
            x=0.5, font=dict(size=14),
        ),
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        legend=dict(
            title="토픽",
            font=dict(size=9),
            itemsizing="constant",
            x=1.01, y=1,
        ),
        width=1300, height=600,
        margin=dict(t=110, b=50, r=240),
        hovermode="closest",
    )

    for col in [1, 2]:
        fig.update_xaxes(range=x_range, showgrid=False, zeroline=False,
                         title_text="UMAP-1", row=1, col=col)
        fig.update_yaxes(range=y_range, showgrid=False, zeroline=False,
                         title_text="UMAP-2" if col == 1 else "", row=1, col=col)

    out_path = DATA_DIR / "bertopic_docs_custom.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
