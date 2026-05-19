#!/usr/bin/env python3
"""
simulation_2 통합 분석
  1. K-means (k=3): 조건별 UMAP 분포 시각화
  2. BERTopic: 기본 시각화 + 조건별 문서 분포

outputs:
  embeddings_cache.npy
  operationalization_k3.json
  kmeans_k3_visualization.html
  bertopic_barchart.html
  bertopic_topics.html
  bertopic_documents.html
  bertopic_condition_dist.html
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
import umap as umap_lib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")
client = OpenAI()

K = 3
PALETTE = ["#E15759", "#4E79A7", "#59A14F"]


# ─── 데이터 로딩 & 파싱 ──────────────────────────────────────────────────────

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


def _split_stmt_sub(text: str) -> tuple[str, str]:
    """첫 문장을 statement, 나머지를 subclaim으로 분리."""
    text = text.strip()
    sp = re.split(r'\n\s+', text, maxsplit=1)
    if len(sp) == 2:
        return sp[0].strip().rstrip('.'), sp[1].strip()
    parts = re.split(r'\.\s+(?=[가-힣A-Z\d])', text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text.rstrip('.'), ""


def parse_response(content):
    content = content.strip()
    points = {}

    # 1) 번호 형식: "1. ..."
    numbered = re.findall(r'(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\Z)', content, re.DOTALL)
    if numbered:
        for num_str, text in numbered:
            num = int(num_str)
            if 1 <= num <= 3:
                stmt, sub = _split_stmt_sub(text)
                points[num] = {"statement": stmt, "subclaim": sub}

    # 2) 불릿 형식: "- ..." 또는 "• ..."
    if not points:
        bullets = re.findall(r'^[-•]\s+(.+?)(?=\n[-•]|\Z)', content, re.DOTALL | re.MULTILINE)
        for i, text in enumerate(bullets[:3], 1):
            stmt, sub = _split_stmt_sub(text)
            points[i] = {"statement": stmt, "subclaim": sub}

    # 3) 단락 형식 (authorless): 빈 줄로 구분된 단락, 첫 단락=주장, 이후=포인트
    if not points:
        paras = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
        for i, text in enumerate(paras[1:4], 1):  # 첫 단락(주장) 건너뜀
            stmt, sub = _split_stmt_sub(text)
            points[i] = {"statement": stmt, "subclaim": sub}

    return points


# ─── 임베딩 (캐시) ───────────────────────────────────────────────────────────

def get_embeddings_cached(texts, cache_path: Path, model="text-embedding-3-small", batch_size=500):
    if cache_path.exists():
        print(f"  임베딩 캐시 로드: {cache_path.name}")
        return np.load(cache_path)
    print(f"  OpenAI 임베딩 계산 중 ({len(texts)}개)...")
    all_embs = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(input=texts[i:i + batch_size], model=model)
        all_embs.extend([e.embedding for e in resp.data])
    embs = np.array(all_embs, dtype=np.float32)
    np.save(cache_path, embs)
    print(f"  캐시 저장: {cache_path.name}")
    return embs


# ─── GPT 레이블 ──────────────────────────────────────────────────────────────

def label_clusters_gpt(cluster_texts: dict[int, list[str]]) -> dict[int, str]:
    prompt_lines = []
    for cid in sorted(cluster_texts):
        examples = cluster_texts[cid][:8]
        prompt_lines.append(f"Cluster {cid}:\n" + "\n".join(f"  - {e}" for e in examples))
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


def label_topics_gpt(topic_info: dict) -> dict[int, str]:
    prompt_lines = []
    for tid, info in topic_info.items():
        if tid == -1:
            continue
        docs = info["representative_docs"][:6]
        prompt_lines.append(f"Topic {tid}:\n" + "\n".join(f"  - {d}" for d in docs))
    if not prompt_lines:
        return {}
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "아래는 동물실험 찬성 논증 문장들의 토픽 클러스터입니다.\n"
                "각 토픽의 핵심 논증 주제를 한국어 10단어 이내로 이름 붙여주세요.\n"
                '응답 형식 (JSON): {"labels": {"0": "주제명", "1": "주제명", ...}}\n\n'
                + "\n\n".join(prompt_lines)
            )
        }],
        response_format={"type": "json_object"},
    )
    raw = json.loads(resp.choices[0].message.content).get("labels", {})
    return {int(k): v for k, v in raw.items()}


def symmetric_kl(t_counts, a_counts):
    t = np.array(t_counts, dtype=float) + 1e-9
    a = np.array(a_counts, dtype=float) + 1e-9
    t /= t.sum(); a /= a.sum()
    return float(np.sum(t * np.log(t / a)) + np.sum(a * np.log(a / t)))


# ─── K-MEANS 분석 ────────────────────────────────────────────────────────────

def run_kmeans(records, embs_norm):
    print("\n" + "=" * 60)
    print(f"K-means (k={K}) 클러스터링")
    print("=" * 60)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(embs_norm)
    for i, rec in enumerate(records):
        rec["cluster"] = int(cluster_ids[i])

    cluster_texts: dict[int, list[str]] = defaultdict(list)
    for rec in records:
        cluster_texts[rec["cluster"]].append(rec["text"])

    print("  GPT 클러스터 레이블링...")
    cluster_labels = label_clusters_gpt(cluster_texts)

    total_t = sum(1 for r in records if r["condition"] == "tangible")
    total_a = sum(1 for r in records if r["condition"] == "authorless")

    cluster_meta: dict[int, dict] = {}
    for cid in range(K):
        members = [r for r in records if r["cluster"] == cid]
        t_cnt   = sum(1 for r in members if r["condition"] == "tangible")
        a_cnt   = sum(1 for r in members if r["condition"] == "authorless")
        cluster_meta[cid] = {
            "label":         cluster_labels.get(cid, f"Cluster {cid}"),
            "total":         len(members),
            "tangible":      t_cnt,
            "authorless":    a_cnt,
            "tangible_pct":  round(100 * t_cnt / len(members), 1) if members else 0,
            "authorless_pct":round(100 * a_cnt / len(members), 1) if members else 0,
            "representative_statements": cluster_texts[cid][:5],
        }

    t_counts = [cluster_meta[c]["tangible"]   for c in range(K)]
    a_counts = [cluster_meta[c]["authorless"]  for c in range(K)]
    kl = symmetric_kl(t_counts, a_counts)

    from sklearn.metrics import silhouette_score
    sil = round(float(silhouette_score(embs_norm, cluster_ids)), 4)

    print(f"\n  Silhouette: {sil}  |  KL divergence: {kl:.4f}")
    for cid in range(K):
        m = cluster_meta[cid]
        print(f"  Cluster {cid} ({m['label']}): n={m['total']}  T={m['tangible_pct']}%  A={m['authorless_pct']}%")

    k3_data = {
        "k": K,
        "silhouette_score": sil,
        "kl_divergence": round(kl, 4),
        "clusters": {str(cid): cluster_meta[cid] for cid in range(K)},
    }
    out_path = DATA_DIR / "operationalization_k3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(k3_data, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_path.name}")

    return cluster_meta, sil, kl, total_t, total_a


def visualize_kmeans(records, embs_norm, cluster_meta, sil, kl, total_t, total_a):
    print("\n  UMAP 2D 계산 중...")
    reducer = umap_lib.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15, metric="cosine")
    coords  = reducer.fit_transform(embs_norm)
    for i, rec in enumerate(records):
        rec["x"] = float(coords[i, 0])
        rec["y"] = float(coords[i, 1])

    labels = {cid: cluster_meta[cid]["label"] for cid in range(K)}

    x_all   = [r["x"] for r in records]
    y_all   = [r["y"] for r in records]
    x_range = [min(x_all) - 0.5, max(x_all) + 0.5]
    y_range = [min(y_all) - 0.5, max(y_all) + 0.5]

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.34, 0.34, 0.32],
        subplot_titles=["Authorless", "Tangible", "클러스터별 조건 분포"],
        horizontal_spacing=0.06,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy", "secondary_y": True}]],
    )

    for col, cond in enumerate(["authorless", "tangible"], start=1):
        subset_all = [r for r in records if r["condition"] == cond]
        other      = [r for r in records if r["condition"] != cond]

        fig.add_trace(go.Scatter(
            x=[r["x"] for r in other],
            y=[r["y"] for r in other],
            mode="markers",
            marker=dict(color="#DDDDDD", size=4, opacity=0.3),
            hoverinfo="skip", showlegend=False, name="",
        ), row=1, col=col)

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
                marker=dict(color=PALETTE[cid], size=7, opacity=0.80,
                            line=dict(width=0.5, color="white")),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=col)

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

        total_cond = len(subset_all)
        cid_counts = {cid: sum(1 for r in subset_all if r["cluster"] == cid) for cid in range(K)}
        dist_text  = "  ".join(
            f"C{cid} {100*cid_counts[cid]/total_cond:.0f}%" if total_cond else f"C{cid} 0%"
            for cid in range(K)
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

    # Panel 3: 조건 분포 비교
    cluster_labels_list = [labels[cid] for cid in range(K)]
    t_within = [100 * cluster_meta[cid]["tangible"]   / total_t for cid in range(K)]
    a_within = [100 * cluster_meta[cid]["authorless"] / total_a for cid in range(K)]

    fig.add_trace(go.Bar(
        name="Tangible",
        x=cluster_labels_list, y=t_within,
        marker_color="#E15759", opacity=0.85,
        text=[f"{v:.1f}%" for v in t_within], textposition="outside",
        hovertemplate="Tangible 내 비율: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=3)

    fig.add_trace(go.Bar(
        name="Authorless",
        x=cluster_labels_list, y=a_within,
        marker_color="#4E79A7", opacity=0.85,
        text=[f"{v:.1f}%" for v in a_within], textposition="outside",
        hovertemplate="Authorless 내 비율: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=3)

    t_in_topic = [cluster_meta[cid]["tangible_pct"] for cid in range(K)]
    fig.add_trace(go.Scatter(
        x=cluster_labels_list, y=t_in_topic,
        mode="markers+text",
        name="토픽 내 Tangible%",
        marker=dict(symbol="diamond", size=11, color="black"),
        text=[f"T내 {v:.0f}%" for v in t_in_topic],
        textposition="top center", textfont=dict(size=9),
        hovertemplate="토픽 내 Tangible 비율: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=3, secondary_y=True)

    expected_t_pct = 100 * total_t / (total_t + total_a)
    fig.add_hline(y=expected_t_pct, line_dash="dot", line_color="gray",
                  line_width=1, row=1, col=3, secondary_y=True)
    fig.add_annotation(
        x=1.0, y=expected_t_pct, xref="x3 domain", yref="y4",
        text=f"기준 {expected_t_pct:.0f}%",
        showarrow=False, font=dict(size=9, color="gray"),
    )

    for i, cid in enumerate(range(K)):
        fig.add_annotation(
            x=cluster_labels_list[i], y=-12, xref="x3", yref="y4",
            text=f"n={cluster_meta[cid]['total']}",
            showarrow=False, font=dict(size=9, color="gray"),
        )

    total_n = total_t + total_a
    fig.update_layout(
        title=dict(
            text=(
                f"K-means Clustering (k={K}) — 조건별 UMAP 2D 비교 [simulation_2]<br>"
                f"<sup>총 {total_n}개 statements  |  "
                f"Silhouette = {sil}  |  KL divergence = {kl}  |  "
                f"tangible {total_t}개 / authorless {total_a}개  |  "
                f"회색 = 반대 조건(참고용)</sup>"
            ),
            x=0.5, font=dict(size=14),
        ),
        barmode="group",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        legend=dict(title="클러스터", font=dict(size=11), itemsizing="constant", x=1.01, y=0.95),
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
    print(f"  저장: {out_path.name}")


# ─── BERTOPIC 분석 ───────────────────────────────────────────────────────────

def run_bertopic(records, texts, embs_norm):
    print("\n" + "=" * 60)
    print("BERTopic 분석")
    print("=" * 60)

    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=True,
    )

    print("\nBERTopic 학습 중...")
    topic_ids, _ = topic_model.fit_transform(texts, embeddings=embs_norm)

    topic_info_df = topic_model.get_topic_info()
    n_topics = len(topic_info_df[topic_info_df["Topic"] != -1])
    n_noise  = sum(1 for t in topic_ids if t == -1)
    print(f"\n  발견된 토픽 수: {n_topics}  (noise: {n_noise}개, {100*n_noise/len(topic_ids):.1f}%)")

    topic_info: dict[int, dict] = {}
    for tid in sorted(set(topic_ids)):
        mask = [i for i, t in enumerate(topic_ids) if t == tid]
        recs = [records[i] for i in mask]
        t_cnt = sum(1 for r in recs if r["condition"] == "tangible")
        a_cnt = sum(1 for r in recs if r["condition"] == "authorless")
        rep_docs = topic_model.get_representative_docs(tid) or [texts[i] for i in mask[:5]]
        topic_info[tid] = {
            "count":               len(mask),
            "tangible":            t_cnt,
            "authorless":          a_cnt,
            "tangible_pct":        round(100 * t_cnt / len(mask), 1) if mask else 0,
            "authorless_pct":      round(100 * a_cnt / len(mask), 1) if mask else 0,
            "representative_docs": rep_docs,
            "keywords":            [w for w, _ in (topic_model.get_topic(tid) or [])[:8]],
        }

    print("\nGPT 토픽 레이블링...")
    gpt_labels = label_topics_gpt(topic_info)
    for tid, label in gpt_labels.items():
        if tid in topic_info:
            topic_info[tid]["label"] = label
    if -1 in topic_info:
        topic_info[-1]["label"] = "Noise (미분류)"

    non_noise = {t: v for t, v in topic_info.items() if t != -1}
    t_counts = [non_noise[t]["tangible"]   for t in sorted(non_noise)]
    a_counts = [non_noise[t]["authorless"] for t in sorted(non_noise)]
    kl = symmetric_kl(t_counts, a_counts)

    expected_t = sum(1 for r in records if r["condition"] == "tangible") / len(records)
    print(f"\n  {'Topic':>6}  {'N':>5}  {'T%':>6}  {'A%':>6}  레이블")
    for tid in sorted(non_noise):
        info = non_noise[tid]
        label = info.get("label", "")
        print(f"  {tid:>6}  {info['count']:>5}  {info['tangible_pct']:>5.1f}%  {info['authorless_pct']:>5.1f}%  {label}")
    print(f"\n  ▶ KL divergence (noise 제외): {kl:.4f}")

    # ── 시각화 1: 문서 scatter ────────────────────────────────────────────────
    print("\n시각화 저장 중...")
    try:
        fig_docs = topic_model.visualize_documents(
            texts, embeddings=embs_norm, custom_labels=True,
            title="BERTopic — Point Statements [simulation_2]",
        )
        fig_docs.write_html(str(DATA_DIR / "bertopic_documents.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_documents.html")
    except Exception as e:
        print(f"  문서 시각화 생략: {e}")

    # ── 시각화 2: 토픽 바 차트 ───────────────────────────────────────────────
    try:
        fig_bar = topic_model.visualize_barchart(top_n_topics=n_topics, n_words=8)
        fig_bar.write_html(str(DATA_DIR / "bertopic_barchart.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_barchart.html")
    except Exception as e:
        print(f"  바 차트 생략: {e}")

    # ── 시각화 3: 인터토픽 거리 맵 ──────────────────────────────────────────
    try:
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html(str(DATA_DIR / "bertopic_topics.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_topics.html")
    except Exception as e:
        print(f"  토픽 맵 생략: {e}")

    # ── 시각화 4: 조건별 문서 UMAP scatter (토픽 색상, 조건별 패널) ────────
    try:
        print("  조건별 UMAP scatter 계산 중...")
        reducer2d = umap_lib.UMAP(n_components=2, random_state=42, min_dist=0.1,
                                   n_neighbors=15, metric="cosine")
        coords2d = reducer2d.fit_transform(embs_norm)

        # topic별 색상 팔레트
        import plotly.express as px
        all_tids = sorted(set(topic_ids))
        color_seq = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        tid_color = {}
        non_noise_tids = [t for t in all_tids if t != -1]
        for i, tid in enumerate(non_noise_tids):
            tid_color[tid] = color_seq[i % len(color_seq)]
        tid_color[-1] = "#CCCCCC"

        fig_cond = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Authorless", "Tangible"],
            horizontal_spacing=0.08,
        )

        x_all_2d = [float(coords2d[i, 0]) for i in range(len(records))]
        y_all_2d = [float(coords2d[i, 1]) for i in range(len(records))]
        x_range2d = [min(x_all_2d) - 0.5, max(x_all_2d) + 0.5]
        y_range2d = [min(y_all_2d) - 0.5, max(y_all_2d) + 0.5]

        for col, cond in enumerate(["authorless", "tangible"], start=1):
            cond_idx   = [i for i, r in enumerate(records) if r["condition"] == cond]
            other_idx  = [i for i, r in enumerate(records) if r["condition"] != cond]

            # 반대 조건 흐리게
            fig_cond.add_trace(go.Scatter(
                x=[x_all_2d[i] for i in other_idx],
                y=[y_all_2d[i] for i in other_idx],
                mode="markers",
                marker=dict(color="#DDDDDD", size=4, opacity=0.25),
                hoverinfo="skip", showlegend=False, name="",
            ), row=1, col=col)

            # 토픽별 점
            for tid in all_tids:
                idx_tid = [i for i in cond_idx if topic_ids[i] == tid]
                if not idx_tid:
                    continue
                label = topic_info[tid].get("label", f"T{tid}")
                hover_list = [
                    f"<b>{label}</b><br>Turn {records[i]['turn']}  Point {records[i]['point']}<br>{records[i]['text']}"
                    for i in idx_tid
                ]
                fig_cond.add_trace(go.Scatter(
                    x=[x_all_2d[i] for i in idx_tid],
                    y=[y_all_2d[i] for i in idx_tid],
                    mode="markers",
                    name=label if tid != -1 else "Noise",
                    legendgroup=f"t{tid}",
                    showlegend=(col == 1),
                    marker=dict(color=tid_color[tid], size=6, opacity=0.8,
                                line=dict(width=0.5, color="white")),
                    text=hover_list,
                    hovertemplate="%{text}<extra></extra>",
                ), row=1, col=col)

            # 토픽 중심 레이블 (noise 제외)
            for tid in non_noise_tids:
                idx_tid = [i for i in cond_idx if topic_ids[i] == tid]
                if not idx_tid:
                    continue
                cx = float(np.mean([x_all_2d[i] for i in idx_tid]))
                cy = float(np.mean([y_all_2d[i] for i in idx_tid]))
                label = topic_info[tid].get("label", f"T{tid}")
                fig_cond.add_annotation(
                    x=cx, y=cy, row=1, col=col,
                    text=f"<b>T{tid}</b>",
                    showarrow=False,
                    font=dict(size=9, color=tid_color[tid]),
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor=tid_color[tid], borderwidth=1, borderpad=2,
                )

            total_cond = len(cond_idx)
            fig_cond.add_annotation(
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.99, y=0.01, row=1, col=col,
                text=f"n={total_cond}",
                showarrow=False,
                font=dict(size=9, color="#555"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="#ccc", borderwidth=1, borderpad=4,
            )

        fig_cond.update_layout(
            title=dict(
                text=f"BERTopic — 조건별 문서 분포 (토픽 색상) [simulation_2]<br>"
                     f"<sup>총 {len(records)}개 statements  |  토픽 수: {n_topics}  |  "
                     f"noise: {n_noise}개  |  KL divergence: {kl:.4f}  |  회색 = 반대 조건</sup>",
                x=0.5, font=dict(size=13),
            ),
            plot_bgcolor="#f9f9f9",
            paper_bgcolor="white",
            legend=dict(title="토픽", font=dict(size=10), x=1.01, y=0.95),
            height=560,
            width=1300,
            margin=dict(t=110, b=60, r=200),
            hovermode="closest",
        )
        for col in [1, 2]:
            fig_cond.update_xaxes(range=x_range2d, showgrid=False, zeroline=False,
                                   title_text="UMAP-1", row=1, col=col)
            fig_cond.update_yaxes(range=y_range2d, showgrid=False, zeroline=False,
                                   title_text="UMAP-2" if col == 1 else "", row=1, col=col)

        fig_cond.write_html(str(DATA_DIR / "bertopic_condition_dist.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_condition_dist.html")
    except Exception as e:
        print(f"  조건별 UMAP 생략: {e}")

    # ── 시각화 5: 토픽별 조건 비율 바 차트 ─────────────────────────────────
    try:
        tids   = sorted(non_noise.keys())
        t_pcts = [non_noise[t]["tangible_pct"]   for t in tids]
        a_pcts = [non_noise[t]["authorless_pct"]  for t in tids]
        xlabels = [f"T{t}: {non_noise[t].get('label','')[:18]}" for t in tids]

        fig_bar2 = go.Figure(data=[
            go.Bar(name="Tangible",   x=xlabels, y=t_pcts, marker_color="#E15759"),
            go.Bar(name="Authorless", x=xlabels, y=a_pcts, marker_color="#4E79A7"),
        ])
        fig_bar2.add_hline(y=expected_t * 100, line_dash="dash", line_color="gray",
                           annotation_text=f"기준선 {expected_t*100:.0f}%")
        fig_bar2.update_layout(
            barmode="group",
            title="토픽별 조건 분포 (Tangible vs Authorless) [simulation_2]",
            yaxis_title="비율 (%)",
            xaxis_title="토픽",
            legend=dict(orientation="h", y=1.1),
        )
        fig_bar2.write_html(str(DATA_DIR / "bertopic_topic_distribution.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_topic_distribution.html")
    except Exception as e:
        print(f"  토픽 분포 바 차트 생략: {e}")

    # ── JSON 저장 ────────────────────────────────────────────────────────────
    output = {
        "n_topics":      n_topics,
        "n_noise":       n_noise,
        "noise_pct":     round(100 * n_noise / len(topic_ids), 1),
        "kl_divergence": round(kl, 4),
        "topics":        {str(t): v for t, v in topic_info.items()},
    }
    with open(DATA_DIR / "bertopic_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("  저장: bertopic_results.json")


# ─── 메인 ────────────────────────────────────────────────────────────────────

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
    total_t = sum(1 for r in records if r["condition"] == "tangible")
    total_a = sum(1 for r in records if r["condition"] == "authorless")
    print(f"  총 {len(texts)}개 statements  (tangible {total_t} / authorless {total_a})")

    embs      = get_embeddings_cached(texts, DATA_DIR / "embeddings_cache.npy")
    embs_norm = normalize(embs)

    # ── K-means ──────────────────────────────────────────────────────────────
    cluster_meta, sil, kl, total_t, total_a = run_kmeans(records, embs_norm)
    visualize_kmeans(records, embs_norm, cluster_meta, sil, kl, total_t, total_a)

    # ── BERTopic ─────────────────────────────────────────────────────────────
    run_bertopic(records, texts, embs_norm)

    print("\n\n분석 완료.")
    print(f"출력 파일:")
    for f in sorted(DATA_DIR.glob("*.html")) :
        print(f"  {f.name}")
    print(f"  operationalization_k3.json")
    print(f"  bertopic_results.json")


if __name__ == "__main__":
    main()
