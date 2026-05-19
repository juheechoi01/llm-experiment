#!/usr/bin/env python3
"""
K-means k=2~6 비교 분석
  - Elbow (inertia), Silhouette, Calinski-Harabasz, Davies-Bouldin
  - KL divergence (조건 간 분포 균형)
  - 각 k별 UMAP scatter (authorless / tangible 패널)
  - 최적 k 권고 요약

outputs:
  kmeans_k_selection.html   — 지표 비교 + UMAP 패널 (인터랙티브)
  kmeans_k_selection.json   — 수치 결과
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap as umap_lib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")

K_RANGE  = range(2, 21)
_BASE_COLORS = [
    "#E15759", "#4E79A7", "#59A14F", "#F28E2B", "#B07AA1",
    "#76B7B2", "#FF9DA7", "#9C755F", "#BAB0AC", "#EDC948",
    "#D4A6C8", "#86BCB6", "#F1CE63", "#A0CBE8", "#FFBE7D",
    "#8CD17D", "#B6992D", "#499894", "#E15ECD", "#79706E",
]
PALETTES = {k: _BASE_COLORS[:k] for k in K_RANGE}


# ─── 파싱 유틸 ───────────────────────────────────────────────────────────────

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
    numbered = re.findall(r'(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\Z)', content, re.DOTALL)
    if numbered:
        for num_str, text in numbered:
            num = int(num_str)
            if 1 <= num <= 3:
                stmt, sub = _split_stmt_sub(text)
                points[num] = {"statement": stmt, "subclaim": sub}
    if not points:
        bullets = re.findall(r'^[-•]\s+(.+?)(?=\n[-•]|\Z)', content, re.DOTALL | re.MULTILINE)
        for i, text in enumerate(bullets[:3], 1):
            stmt, sub = _split_stmt_sub(text)
            points[i] = {"statement": stmt, "subclaim": sub}
    if not points:
        paras = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
        for i, text in enumerate(paras[1:4], 1):
            stmt, sub = _split_stmt_sub(text)
            points[i] = {"statement": stmt, "subclaim": sub}
    return points


def symmetric_kl(t_counts, a_counts):
    t = np.array(t_counts, dtype=float) + 1e-9
    a = np.array(a_counts, dtype=float) + 1e-9
    t /= t.sum(); a /= a.sum()
    return float(np.sum(t * np.log(t / a)) + np.sum(a * np.log(a / t)))


# ─── 클러스터링 & 지표 계산 ──────────────────────────────────────────────────

def run_all_k(records, embs_norm, coords2d):
    results = {}
    for k in K_RANGE:
        print(f"  k={k} 클러스터링...")
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(embs_norm)

        sil = float(silhouette_score(embs_norm, labels))
        ch  = float(calinski_harabasz_score(embs_norm, labels))
        db  = float(davies_bouldin_score(embs_norm, labels))
        inertia = float(km.inertia_)

        total_t = sum(1 for r in records if r["condition"] == "tangible")
        total_a = sum(1 for r in records if r["condition"] == "authorless")

        clusters = {}
        for cid in range(k):
            idx     = [i for i, l in enumerate(labels) if l == cid]
            t_cnt   = sum(1 for i in idx if records[i]["condition"] == "tangible")
            a_cnt   = sum(1 for i in idx if records[i]["condition"] == "authorless")
            clusters[cid] = {
                "n":            len(idx),
                "tangible":     t_cnt,
                "authorless":   a_cnt,
                "tangible_pct": round(100 * t_cnt / len(idx), 1) if idx else 0,
            }

        t_counts = [clusters[c]["tangible"]   for c in range(k)]
        a_counts = [clusters[c]["authorless"] for c in range(k)]
        kl = symmetric_kl(t_counts, a_counts)

        results[k] = {
            "silhouette":         round(sil, 4),
            "calinski_harabasz":  round(ch, 2),
            "davies_bouldin":     round(db, 4),
            "inertia":            round(inertia, 2),
            "kl_divergence":      round(kl, 4),
            "cluster_labels":     labels.tolist(),
            "clusters":           clusters,
        }
        print(f"     sil={sil:.4f}  CH={ch:.1f}  DB={db:.4f}  inertia={inertia:.1f}  KL={kl:.4f}")

    return results


# ─── 지표별 best k & 권고 계산 ──────────────────────────────────────────────

def compute_recommendation(results):
    ks = list(results.keys())
    sils     = {k: results[k]["silhouette"]        for k in ks}
    chs      = {k: results[k]["calinski_harabasz"] for k in ks}
    dbs      = {k: results[k]["davies_bouldin"]    for k in ks}
    inertias = {k: results[k]["inertia"]           for k in ks}

    best_sil   = max(sils, key=sils.get)
    best_ch    = max(chs,  key=chs.get)
    best_db    = min(dbs,  key=dbs.get)

    inertia_vals = [inertias[k] for k in ks]
    if len(inertia_vals) >= 3:
        d2 = np.diff(np.diff(inertia_vals))
        best_elbow = ks[int(np.argmax(d2)) + 1]
    else:
        best_elbow = ks[0]

    votes = defaultdict(int)
    for best in [best_sil, best_ch, best_db, best_elbow]:
        votes[best] += 1
    recommended_k = max(votes, key=lambda k: (votes[k], -k))

    return {
        "best_sil": best_sil, "best_ch": best_ch,
        "best_db": best_db,   "best_elbow": best_elbow,
        "recommended_k": recommended_k, "votes": dict(votes),
        "sils": sils, "chs": chs, "dbs": dbs, "inertias": inertias,
    }


# ─── 시각화 ──────────────────────────────────────────────────────────────────

def build_figure(records, coords2d, results):
    ks  = list(results.keys())
    rec = compute_recommendation(results)

    # UMAP scatter에 사용할 후보 k: 각 지표 best + 권고 k, 정렬·중복 제거, 최대 6개
    candidate_ks = sorted(set([
        rec["best_sil"], rec["best_ch"], rec["best_db"],
        rec["best_elbow"], rec["recommended_k"],
    ]))[:6]
    n_cand = len(candidate_ks)

    # ── 레이아웃 설계 ─────────────────────────────────────────────────────────
    # 행 1~2 : 지표 선 그래프 (2행 × 4열)
    # 행 3   : KL divergence (1행 × 4열, 전체 폭)
    # 행 4~  : UMAP scatter  (후보 k별 1행 × 4열: auth | tang | bar(colspan=2))
    N_METRIC = 3
    total_rows = N_METRIC + n_cand

    row_heights = [0.13, 0.13, 0.08] + [0.66 / n_cand] * n_cand

    specs = (
        [[{"type": "xy"}] * 4] * 2              # 행 1~2: 지표
        + [[{"type": "xy", "colspan": 4}, None, None, None]]  # 행 3: KL
        + [[{"type": "xy"}, {"type": "xy"}, {"type": "xy", "colspan": 2}, None]] * n_cand  # UMAP
    )

    metric_titles = ["Silhouette ↑", "Calinski-Harabasz ↑", "Davies-Bouldin ↓", "Inertia (Elbow) ↓"]
    kl_title      = ["KL Divergence (조건 간 분포 균형, 낮을수록 균등)"]
    umap_titles   = []
    for k in candidate_ks:
        tag = " ★" if k == rec["recommended_k"] else ""
        umap_titles += [f"k={k}{tag}  Authorless", f"k={k}{tag}  Tangible", f"k={k}{tag}  클러스터별 T%", ""]

    subplot_titles = metric_titles + kl_title + umap_titles

    fig = make_subplots(
        rows=total_rows, cols=4,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        horizontal_spacing=0.06,
        vertical_spacing=0.045,
        specs=specs,
    )

    # ── 행 1~2: 지표 선 그래프 ───────────────────────────────────────────────
    metric_cfg = [
        ("silhouette",        "Silhouette",        "#2CA02C", True,  1, 1),
        ("calinski_harabasz", "Calinski-Harabasz", "#1F77B4", True,  1, 2),
        ("davies_bouldin",    "Davies-Bouldin",    "#D62728", False, 1, 3),
        ("inertia",           "Inertia",           "#FF7F0E", False, 1, 4),
    ]

    for mkey, mname, mcolor, higher_better, mrow, mcol in metric_cfg:
        yvals  = [results[k][mkey] for k in ks]
        best_k = ks[int(np.argmax(yvals))] if higher_better else ks[int(np.argmin(yvals))]

        fig.add_trace(go.Scatter(
            x=ks, y=yvals,
            mode="lines+markers",
            name=mname,
            line=dict(color=mcolor, width=2),
            marker=dict(
                size=[10 if k == best_k else 6 for k in ks],
                color=["gold" if k == best_k else mcolor for k in ks],
                symbol=["star" if k == best_k else "circle" for k in ks],
                line=dict(width=1.5, color="black"),
            ),
            showlegend=False,
            hovertemplate=f"k=%{{x}}<br>{mname}=%{{y:.4f}}<extra></extra>",
        ), row=mrow, col=mcol)

        # 권고 k 수직선
        fig.add_vline(x=rec["recommended_k"], line_dash="dash",
                      line_color="rgba(150,0,200,0.4)", line_width=1.5,
                      row=mrow, col=mcol)
        fig.add_annotation(
            x=best_k, y=results[best_k][mkey],
            text=f"★k={best_k}",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            font=dict(size=8, color="black"),
            bgcolor="rgba(255,255,0,0.75)", borderpad=2,
            row=mrow, col=mcol,
        )
        fig.update_xaxes(tickvals=ks, tickfont=dict(size=9), title_text="k", row=mrow, col=mcol)
        fig.update_yaxes(title_text=mname, title_font=dict(size=9), row=mrow, col=mcol)

    # ── 행 3: KL divergence ──────────────────────────────────────────────────
    kl_vals = [results[k]["kl_divergence"] for k in ks]
    kl_colors = ["#D62728" if v > 0.15 else ("#FF7F0E" if v > 0.05 else "#2CA02C") for v in kl_vals]

    fig.add_trace(go.Bar(
        x=ks, y=kl_vals,
        marker_color=kl_colors,
        opacity=0.8,
        showlegend=False,
        text=[f"{v:.3f}" for v in kl_vals],
        textposition="outside",
        textfont=dict(size=8),
        hovertemplate="k=%{x}<br>KL=%{y:.4f}<extra></extra>",
    ), row=3, col=1)

    # 기준선
    for thresh, label, dash in [(0.05, "낮음(0.05)", "dot"), (0.15, "중간(0.15)", "dash")]:
        fig.add_hline(y=thresh, line_dash=dash, line_color="gray", line_width=1,
                      row=3, col=1)
        fig.add_annotation(
            x=max(ks) + 0.3, y=thresh, xref="x5", yref="y5",
            text=label, showarrow=False, font=dict(size=8, color="gray"),
        )
    fig.add_vline(x=rec["recommended_k"], line_dash="dash",
                  line_color="rgba(150,0,200,0.4)", line_width=1.5, row=3, col=1)
    fig.update_xaxes(tickvals=ks, tickfont=dict(size=9), title_text="k", row=3, col=1)
    fig.update_yaxes(title_text="Symmetric KL", title_font=dict(size=9), row=3, col=1)

    # ── 행 4+: UMAP scatter (후보 k만) ──────────────────────────────────────
    x_all   = [float(coords2d[i, 0]) for i in range(len(records))]
    y_all   = [float(coords2d[i, 1]) for i in range(len(records))]
    x_range = [min(x_all) - 0.5, max(x_all) + 0.5]
    y_range = [min(y_all) - 0.5, max(y_all) + 0.5]

    for row_offset, k in enumerate(candidate_ks):
        row     = N_METRIC + 1 + row_offset
        palette = PALETTES[k]
        labels  = results[k]["cluster_labels"]

        for col, cond in enumerate(["authorless", "tangible"], start=1):
            cond_idx  = [i for i, r in enumerate(records) if r["condition"] == cond]
            other_idx = [i for i, r in enumerate(records) if r["condition"] != cond]

            fig.add_trace(go.Scatter(
                x=[x_all[i] for i in other_idx],
                y=[y_all[i] for i in other_idx],
                mode="markers",
                marker=dict(color="#DDDDDD", size=3, opacity=0.2),
                hoverinfo="skip", showlegend=False, name="",
            ), row=row, col=col)

            for cid in range(k):
                idx_cid = [i for i in cond_idx if labels[i] == cid]
                if not idx_cid:
                    continue
                n_cid = results[k]["clusters"][cid]["n"]
                t_pct = results[k]["clusters"][cid]["tangible_pct"]
                hover = [
                    f"<b>C{cid}</b> n={n_cid} T={t_pct:.0f}%<br>{records[i]['text'][:70]}"
                    for i in idx_cid
                ]
                fig.add_trace(go.Scatter(
                    x=[x_all[i] for i in idx_cid],
                    y=[y_all[i] for i in idx_cid],
                    mode="markers",
                    name=f"C{cid}",
                    legendgroup=f"k{k}c{cid}",
                    showlegend=False,
                    marker=dict(color=palette[cid], size=4.5, opacity=0.75,
                                line=dict(width=0.3, color="white")),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                ), row=row, col=col)

            # 클러스터 중심 레이블
            for cid in range(k):
                idx_cid = [i for i in cond_idx if labels[i] == cid]
                if not idx_cid:
                    continue
                cx = float(np.mean([x_all[i] for i in idx_cid]))
                cy = float(np.mean([y_all[i] for i in idx_cid]))
                t_pct = results[k]["clusters"][cid]["tangible_pct"]
                fig.add_annotation(
                    x=cx, y=cy, row=row, col=col,
                    text=f"<b>C{cid}</b> T{t_pct:.0f}%",
                    showarrow=False,
                    font=dict(size=7.5, color=palette[cid]),
                    bgcolor="rgba(255,255,255,0.78)",
                    bordercolor=palette[cid], borderwidth=1, borderpad=2,
                )

        # 조건 분포 막대 (col=3, colspan=2)
        cids    = list(range(k))
        t_pcts  = [results[k]["clusters"][c]["tangible_pct"] for c in cids]
        ns      = [results[k]["clusters"][c]["n"] for c in cids]
        xlabels = [f"C{c}(n={ns[c]})" for c in cids]

        fig.add_trace(go.Bar(
            x=xlabels, y=t_pcts,
            name="T%",
            marker_color=palette,
            opacity=0.85, showlegend=False,
            text=[f"{v:.0f}%" for v in t_pcts],
            textposition="inside",
            textfont=dict(size=8, color="white"),
            hovertemplate="C%{x}: T %{y:.1f}%<extra></extra>",
        ), row=row, col=3)

        fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1, row=row, col=3)
        fig.update_yaxes(range=[0, 115], title_text="Tangible %",
                         title_font=dict(size=8), row=row, col=3)
        fig.update_xaxes(tickfont=dict(size=7.5), row=row, col=3)

        kl_k  = results[k]["kl_divergence"]
        sil_k = results[k]["silhouette"]
        tag   = "  ★권고" if k == rec["recommended_k"] else ""
        fig.add_annotation(
            x=0.98, y=0.97,
            xref=f"x{(N_METRIC + 1 + row_offset - 1)*3 + 3 + 1} domain"
                 if (N_METRIC + 1 + row_offset) > 1 else "x3 domain",
            yref=f"y{(N_METRIC + 1 + row_offset - 1)*3 + 3 + 1} domain"
                 if (N_METRIC + 1 + row_offset) > 1 else "y3 domain",
            row=row, col=3,
            text=f"sil={sil_k}  KL={kl_k}{tag}",
            showarrow=False,
            font=dict(size=9, color="#222"),
            bgcolor="rgba(255,255,240,0.88)",
            bordercolor="#bbb", borderwidth=1, borderpad=3,
            align="right",
        )

        for col in [1, 2]:
            fig.update_xaxes(range=x_range, showgrid=False, zeroline=False,
                             showticklabels=False, row=row, col=col)
            fig.update_yaxes(range=y_range, showgrid=False, zeroline=False,
                             showticklabels=False, row=row, col=col)

    # ── 요약 & 전체 레이아웃 ─────────────────────────────────────────────────
    summary = (
        f"<b>지표별 권고 k</b>　　"
        f"Silhouette ↑: k=<b>{rec['best_sil']}</b> ({rec['sils'][rec['best_sil']]:.4f})　　"
        f"Calinski-Harabasz ↑: k=<b>{rec['best_ch']}</b> ({rec['chs'][rec['best_ch']]:.1f})　　"
        f"Davies-Bouldin ↓: k=<b>{rec['best_db']}</b> ({rec['dbs'][rec['best_db']]:.4f})　　"
        f"Elbow: k=<b>{rec['best_elbow']}</b>　　"
        f"　🏆 종합 권고: k=<b>{rec['recommended_k']}</b>　　"
        f"(UMAP 표시 후보: k={candidate_ks})"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"K-means k=2~20 비교 분석 [simulation_2]<br>"
                f"<sup>{summary}</sup>"
            ),
            x=0.5, font=dict(size=13),
        ),
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        height=480 + 300 * n_cand,
        width=1400,
        margin=dict(t=130, b=60, r=60),
        hovermode="closest",
        barmode="relative",
    )

    return fig, rec["recommended_k"], rec["votes"], candidate_ks


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
    print(f"  총 {len(texts)}개 statements")

    cache_path = DATA_DIR / "embeddings_cache.npy"
    if not cache_path.exists():
        print("  임베딩 캐시 없음 — analyze.py를 먼저 실행하세요.")
        return
    embs      = np.load(cache_path)
    embs_norm = normalize(embs)
    print(f"  임베딩 캐시 로드: {cache_path.name}")

    print("\nUMAP 2D 계산 중...")
    reducer = umap_lib.UMAP(n_components=2, random_state=42, min_dist=0.1,
                             n_neighbors=15, metric="cosine")
    coords2d = reducer.fit_transform(embs_norm)
    print("  완료")

    print("\nK-means k=2~20 클러스터링 & 지표 계산...")
    results = run_all_k(records, embs_norm, coords2d)

    print("\n시각화 생성 중...")
    fig, recommended_k, votes, candidate_ks = build_figure(records, coords2d, results)

    out_html = DATA_DIR / "kmeans_k_selection.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"  저장: {out_html.name}")

    # JSON 저장
    out_json = DATA_DIR / "kmeans_k_selection.json"
    save_data = {
        k: {
            mk: mv for mk, mv in v.items() if mk != "cluster_labels"  # labels는 용량 큼
        }
        for k, v in results.items()
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"k_range": list(K_RANGE), "results": {str(k): v for k, v in save_data.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_json.name}")

    print("\n" + "=" * 65)
    print("지표별 권고 요약")
    print("=" * 65)
    print(f"  {'k':>3}  {'Silhouette':>10}  {'CH':>10}  {'DB':>8}  {'Inertia':>10}  {'KL':>8}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    for k in K_RANGE:
        r   = results[k]
        tag = "  ← 권고" if k == recommended_k else ""
        print(f"  {k:>3}  {r['silhouette']:>10.4f}  {r['calinski_harabasz']:>10.1f}"
              f"  {r['davies_bouldin']:>8.4f}  {r['inertia']:>10.1f}  {r['kl_divergence']:>8.4f}{tag}")
    print(f"\n  투표 결과: {dict(votes)}")
    print(f"  UMAP 후보: k={candidate_ks}")
    print(f"  → 종합 권고: k={recommended_k}")
    print("\n완료.")


if __name__ == "__main__":
    main()
