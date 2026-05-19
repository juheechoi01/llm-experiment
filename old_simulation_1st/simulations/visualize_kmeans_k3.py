#!/usr/bin/env python3
"""
K-means (k=3) 클러스터 시각화
  - Panel 1: Authorless UMAP (tangible 회색 배경)
  - Panel 2: Tangible UMAP  (authorless 회색 배경)
  - Panel 3: 클러스터별 조건 분포 비교
  - 결과: kmeans_k3_visualization.html
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")

K       = 3
PALETTE = ["#E15759", "#4E79A7", "#59A14F"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_assistant_turns(sims):
    turns = []
    for sim in sims:
        for t in sim["conversation"]:
            if t["role"] == "assistant":
                turns.append({
                    "sim_id": sim["sim_id"], "condition": sim["condition"],
                    "turn": t["turn"], "content": t["content"],
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
        for i, text in enumerate(
            re.findall(r'^[-•]\s+(.+?)(?=\n[-•]|\Z)', content, re.DOTALL | re.MULTILINE)[:3], 1
        ):
            text = text.strip()
            points[i] = {"statement": text.split('\n')[0].strip().rstrip('.'), "subclaim": ""}
    return points


def main():
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

    embs      = np.load(DATA_DIR / "embeddings_cache.npy")
    embs_norm = normalize(embs)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(embs_norm)
    for i, rec in enumerate(records):
        rec["cluster"] = int(cluster_ids[i])

    with open(DATA_DIR / "operationalization_k3.json", encoding="utf-8") as f:
        k3_data = json.load(f)
    cluster_meta = {int(k): v for k, v in k3_data["clusters"].items()}
    labels = {cid: cluster_meta[cid]["label"] for cid in range(K)}

    print("UMAP 2D 계산 중...")
    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15, metric="cosine")
    coords  = reducer.fit_transform(embs_norm)
    for i, rec in enumerate(records):
        rec["x"] = float(coords[i, 0])
        rec["y"] = float(coords[i, 1])

    x_all   = [r["x"] for r in records]
    y_all   = [r["y"] for r in records]
    x_range = [min(x_all) - 0.5, max(x_all) + 0.5]
    y_range = [min(y_all) - 0.5, max(y_all) + 0.5]

    total_t = sum(1 for r in records if r["condition"] == "tangible")
    total_a = sum(1 for r in records if r["condition"] == "authorless")

    # ── Figure: 1행 3열 ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.34, 0.34, 0.32],
        subplot_titles=["Authorless", "Tangible", "클러스터별 조건 분포"],
        horizontal_spacing=0.06,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy", "secondary_y": True}]],
    )

    # ── Panel 1 & 2: 조건별 UMAP scatter ────────────────────────────────────
    for col, cond in enumerate(["authorless", "tangible"], start=1):
        subset_all = [r for r in records if r["condition"] == cond]
        other      = [r for r in records if r["condition"] != cond]

        # 반대 조건 흐리게
        fig.add_trace(go.Scatter(
            x=[r["x"] for r in other],
            y=[r["y"] for r in other],
            mode="markers",
            marker=dict(color="#DDDDDD", size=4, opacity=0.3),
            hoverinfo="skip", showlegend=False, name="",
        ), row=1, col=col)

        # 클러스터별 색상
        for cid in range(K):
            subset = [r for r in subset_all if r["cluster"] == cid]
            if not subset:
                continue
            hover = [
                f"<b>Cluster {cid}: {labels[cid]}</b><br>"
                f"Turn {r['turn']}  |  Point {r['point']}<br><br>"
                f"<b>Statement:</b> {r['text']}<br>"
                f"<b>Sub-claim:</b> {r['subclaim'][:100]}{'…' if len(r['subclaim'])>100 else ''}"
                for r in subset
            ]
            fig.add_trace(go.Scatter(
                x=[r["x"] for r in subset],
                y=[r["y"] for r in subset],
                mode="markers",
                name=f"C{cid}: {labels[cid]}",
                legendgroup=f"c{cid}",
                showlegend=(col == 1),
                marker=dict(
                    color=PALETTE[cid], size=7, opacity=0.80,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=col)

        # 클러스터 중심 레이블
        for cid in range(K):
            sub = [r for r in subset_all if r["cluster"] == cid]
            if not sub:
                continue
            cx = float(np.mean([r["x"] for r in sub]))
            cy = float(np.mean([r["y"] for r in sub]))
            fig.add_annotation(
                x=cx, y=cy, row=1, col=col,
                text=f"<b>C{cid}</b><br>{labels[cid]}",
                showarrow=False,
                font=dict(size=9, color=PALETTE[cid]),
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor=PALETTE[cid], borderwidth=1.5, borderpad=3,
            )

        # 조건별 클러스터 비율 (우하단)
        total_cond = len(subset_all)
        cid_counts = {cid: sum(1 for r in subset_all if r["cluster"] == cid) for cid in range(K)}
        dist_text  = "  ".join(
            f"C{cid} {100*cid_counts[cid]/total_cond:.0f}%" for cid in range(K)
        )
        xref_str = f"x{col} domain" if col > 1 else "x domain"
        yref_str = f"y{col} domain" if col > 1 else "y domain"
        fig.add_annotation(
            xref=xref_str, yref=yref_str,
            x=0.99, y=0.01, row=1, col=col,
            text=f"n={total_cond}<br>{dist_text}",
            showarrow=False,
            font=dict(size=9, color="#555"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#ccc", borderwidth=1, borderpad=4,
            align="right",
        )

    # ── Panel 3: 조건 분포 비교 ──────────────────────────────────────────────
    cluster_labels = [labels[cid] for cid in range(K)]

    # 각 조건 내 토픽 분포 (within-condition %)
    t_within = [100 * cluster_meta[cid]["tangible"]   / total_t for cid in range(K)]
    a_within = [100 * cluster_meta[cid]["authorless"] / total_a for cid in range(K)]

    fig.add_trace(go.Bar(
        name="Tangible (조건 내 %)",
        x=cluster_labels, y=t_within,
        marker_color="#E15759", opacity=0.85,
        text=[f"{v:.1f}%" for v in t_within], textposition="outside",
        hovertemplate="Tangible 내 비율: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=3)

    fig.add_trace(go.Bar(
        name="Authorless (조건 내 %)",
        x=cluster_labels, y=a_within,
        marker_color="#4E79A7", opacity=0.85,
        text=[f"{v:.1f}%" for v in a_within], textposition="outside",
        hovertemplate="Authorless 내 비율: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=3)

    # 토픽 내 Tangible% (보조 y축, 마름모)
    t_in_topic = [cluster_meta[cid]["tangible_pct"] for cid in range(K)]
    fig.add_trace(go.Scatter(
        x=cluster_labels, y=t_in_topic,
        mode="markers+text",
        name="토픽 내 Tangible%",
        marker=dict(symbol="diamond", size=11, color="black"),
        text=[f"T내 {v:.0f}%" for v in t_in_topic],
        textposition="top center", textfont=dict(size=9),
        hovertemplate="토픽 내 Tangible 비율: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=3, secondary_y=True)

    # 기준선 annotation (토픽 내 T% 기준)
    expected_t_pct = 100 * total_t / (total_t + total_a)
    fig.add_annotation(
        x=1.0, y=expected_t_pct, xref="x3 domain", yref="y4",
        text=f"기준 {expected_t_pct:.0f}%",
        showarrow=False, font=dict(size=9, color="gray"),
    )
    fig.add_hline(y=expected_t_pct, line_dash="dot", line_color="gray",
                  line_width=1, row=1, col=3, secondary_y=True)

    # 클러스터 크기 annotation
    for i, cid in enumerate(range(K)):
        fig.add_annotation(
            x=cluster_labels[i], y=-12, xref="x3", yref="y4",
            text=f"n={cluster_meta[cid]['total']}",
            showarrow=False, font=dict(size=9, color="gray"),
        )

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    sil = k3_data["silhouette_score"]
    kl  = k3_data["kl_divergence"]

    fig.update_layout(
        title=dict(
            text=(
                f"K-means Clustering (k=3) — 조건별 UMAP 2D 비교<br>"
                f"<sup>총 {len(records)}개 statements  |  "
                f"Silhouette = {sil}  |  KL divergence = {kl}  |  "
                f"tangible {total_t}개 / authorless {total_a}개  |  "
                f"회색 = 반대 조건(참고용)</sup>"
            ),
            x=0.5, font=dict(size=14),
        ),
        barmode="group",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        legend=dict(
            title="클러스터",
            font=dict(size=11),
            itemsizing="constant",
            x=1.01, y=0.95,
        ),
        height=580,
        width=1400,
        margin=dict(t=110, b=60, r=160),
        hovermode="closest",
    )

    for col in [1, 2]:
        fig.update_xaxes(range=x_range, showgrid=False, zeroline=False,
                         title_text="UMAP-1", row=1, col=col)
        fig.update_yaxes(range=y_range, showgrid=False, zeroline=False,
                         title_text="UMAP-2" if col == 1 else "", row=1, col=col)

    fig.update_xaxes(tickangle=-15, tickfont=dict(size=9), row=1, col=3)
    fig.update_yaxes(title_text="조건 내 비율 (%)", range=[0, 80],
                     row=1, col=3, secondary_y=False)
    fig.update_yaxes(title_text="토픽 내 Tangible %", range=[0, 100],
                     showgrid=False, tickfont=dict(size=9),
                     row=1, col=3, secondary_y=True)

    out_path = DATA_DIR / "kmeans_k3_visualization.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
