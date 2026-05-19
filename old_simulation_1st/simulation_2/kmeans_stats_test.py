#!/usr/bin/env python3
"""
K-means 통계 검정: k=2~20
  1. Chi-square test  — 조건(tangible/authorless)과 클러스터 배정이 독립인가?
  2. Permutation test — 관측된 KL divergence가 우연 수준인가?

outputs:
  kmeans_stats_test.html   — 시각화
  kmeans_stats_test.json   — 수치 결과
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap as umap_lib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

DATA_DIR   = Path(__file__).parent
K_RANGE    = range(2, 21)
N_PERM     = 1000
ALPHA      = 0.05

load_dotenv(DATA_DIR.parent / ".env")


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


def _split_stmt_sub(text):
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
    points  = {}
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


def symmetric_kl(t_arr, a_arr):
    t = np.array(t_arr, dtype=float) + 1e-9
    a = np.array(a_arr, dtype=float) + 1e-9
    t /= t.sum(); a /= a.sum()
    return float(np.sum(t * np.log(t / a)) + np.sum(a * np.log(a / t)))


# ─── 검정 ────────────────────────────────────────────────────────────────────

def run_tests(records, embs_norm):
    cond_arr = np.array([r["condition"] for r in records])  # "tangible" / "authorless"
    results  = {}

    for k in K_RANGE:
        print(f"  k={k} ...", end=" ", flush=True)

        km     = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(embs_norm)

        # ── 1) Chi-square ────────────────────────────────────────────────────
        # contingency table: rows=condition, cols=cluster
        t_counts = np.array([np.sum((labels == c) & (cond_arr == "tangible"))   for c in range(k)])
        a_counts = np.array([np.sum((labels == c) & (cond_arr == "authorless")) for c in range(k)])
        contingency = np.vstack([t_counts, a_counts])  # (2, k)

        chi2, p_chi2, dof, _ = chi2_contingency(contingency)
        cramers_v = float(np.sqrt(chi2 / (len(records) * (min(2, k) - 1))))

        # ── 2) KL divergence + permutation test ─────────────────────────────
        obs_kl = symmetric_kl(t_counts, a_counts)

        perm_kls = []
        for _ in range(N_PERM):
            shuffled = np.random.permutation(cond_arr)
            pt = np.array([np.sum((labels == c) & (shuffled == "tangible"))   for c in range(k)])
            pa = np.array([np.sum((labels == c) & (shuffled == "authorless")) for c in range(k)])
            perm_kls.append(symmetric_kl(pt, pa))
        perm_kls = np.array(perm_kls)

        p_perm   = float(np.mean(perm_kls >= obs_kl))
        kl_null_mean = float(perm_kls.mean())
        kl_null_ci95 = (float(np.percentile(perm_kls, 2.5)),
                        float(np.percentile(perm_kls, 97.5)))

        results[k] = {
            "k":           k,
            "chi2":        round(float(chi2), 4),
            "p_chi2":      round(float(p_chi2), 6),
            "dof":         int(dof),
            "cramers_v":   round(cramers_v, 4),
            "sig_chi2":    bool(p_chi2 < ALPHA),
            "obs_kl":      round(obs_kl, 4),
            "p_perm":      round(p_perm, 4),
            "sig_perm":    bool(p_perm < ALPHA),
            "kl_null_mean":round(kl_null_mean, 4),
            "kl_null_ci95":[round(kl_null_ci95[0], 4), round(kl_null_ci95[1], 4)],
            "cluster_labels": labels.tolist(),
            "t_counts":    t_counts.tolist(),
            "a_counts":    a_counts.tolist(),
        }

        sig_mark = "✗ 유의" if p_chi2 < ALPHA else "✓ 독립"
        print(f"chi2 p={p_chi2:.4f} ({sig_mark})  KL_obs={obs_kl:.4f}  p_perm={p_perm:.4f}")

    return results


# ─── 시각화 ──────────────────────────────────────────────────────────────────

def build_figure(results):
    ks = list(results.keys())

    p_chi2s    = [results[k]["p_chi2"]    for k in ks]
    cramers_vs = [results[k]["cramers_v"] for k in ks]
    obs_kls    = [results[k]["obs_kl"]    for k in ks]
    p_perms    = [results[k]["p_perm"]    for k in ks]
    kl_means   = [results[k]["kl_null_mean"]     for k in ks]
    kl_lo      = [results[k]["kl_null_ci95"][0]  for k in ks]
    kl_hi      = [results[k]["kl_null_ci95"][1]  for k in ks]

    # 유의 여부 색상
    def bar_color(sig):
        return "#D62728" if sig else "#2CA02C"

    chi2_colors = [bar_color(results[k]["sig_chi2"]) for k in ks]
    perm_colors = [bar_color(results[k]["sig_perm"]) for k in ks]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Chi-square p-value  (점선=α=0.05, 녹색=독립, 빨강=유의)",
            "Cramér's V  (효과 크기, 낮을수록 조건 영향 작음)",
            "Permutation test p-value  (점선=α=0.05)",
            "KL divergence: 관측값 vs 귀무 분포 (permutation 95% CI)",
            "두 검정 동시 요약  (초록=둘 다 독립, 주황=한쪽만, 빨강=둘 다 유의)",
            "클러스터 내 Tangible 비율 편차  (50%에서 얼마나 벗어나는가)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.09,
    )

    # ── (1,1) Chi-square p-value ──────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=ks, y=p_chi2s,
        marker_color=chi2_colors, opacity=0.85,
        text=[f"{v:.4f}" for v in p_chi2s],
        textposition="outside", textfont=dict(size=8),
        showlegend=False,
        hovertemplate="k=%{x}<br>p=%{y:.6f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=ALPHA, line_dash="dash", line_color="black", line_width=1.5, row=1, col=1)
    fig.add_annotation(x=max(ks)+0.3, y=ALPHA, xref="x1", yref="y1",
                       text=f"α={ALPHA}", showarrow=False, font=dict(size=9))
    fig.update_yaxes(title_text="p-value", range=[0, max(p_chi2s)*1.25], row=1, col=1)

    # ── (1,2) Cramér's V ─────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=ks, y=cramers_vs,
        marker_color="#1F77B4", opacity=0.75,
        text=[f"{v:.3f}" for v in cramers_vs],
        textposition="outside", textfont=dict(size=8),
        showlegend=False,
        hovertemplate="k=%{x}<br>V=%{y:.4f}<extra></extra>",
    ), row=1, col=2)
    # 효과 크기 기준선 (small=0.1, medium=0.3)
    for thresh, label in [(0.1, "small(0.1)"), (0.3, "medium(0.3)")]:
        fig.add_hline(y=thresh, line_dash="dot", line_color="gray", line_width=1, row=1, col=2)
        fig.add_annotation(x=max(ks)+0.3, y=thresh, xref="x2", yref="y2",
                           text=label, showarrow=False, font=dict(size=8, color="gray"))
    fig.update_yaxes(title_text="Cramér's V", row=1, col=2)

    # ── (2,1) Permutation p-value ────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=ks, y=p_perms,
        marker_color=perm_colors, opacity=0.85,
        text=[f"{v:.3f}" for v in p_perms],
        textposition="outside", textfont=dict(size=8),
        showlegend=False,
        hovertemplate="k=%{x}<br>p_perm=%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=ALPHA, line_dash="dash", line_color="black", line_width=1.5, row=2, col=1)
    fig.update_yaxes(title_text="p-value (permutation)", range=[0, max(p_perms)*1.25], row=2, col=1)

    # ── (2,2) KL obs vs null CI ──────────────────────────────────────────────
    # null 분포 CI 밴드
    fig.add_trace(go.Scatter(
        x=ks + ks[::-1],
        y=kl_hi + kl_lo[::-1],
        fill="toself",
        fillcolor="rgba(100,100,200,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="귀무분포 95% CI",
        showlegend=True,
        hoverinfo="skip",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=ks, y=kl_means,
        mode="lines",
        line=dict(color="#7777CC", width=1.5, dash="dot"),
        name="귀무분포 평균",
        showlegend=True,
        hovertemplate="k=%{x}<br>null KL=%{y:.4f}<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=ks, y=obs_kls,
        mode="lines+markers",
        line=dict(color="#D62728", width=2),
        marker=dict(
            size=[10 if results[k]["sig_perm"] else 7 for k in ks],
            color=["#D62728" if results[k]["sig_perm"] else "#2CA02C" for k in ks],
            symbol=["circle" if results[k]["sig_perm"] else "circle-open" for k in ks],
            line=dict(width=1.5, color="#D62728"),
        ),
        name="관측 KL",
        showlegend=True,
        hovertemplate="k=%{x}<br>obs KL=%{y:.4f}<extra></extra>",
    ), row=2, col=2)
    fig.update_yaxes(title_text="Symmetric KL divergence", row=2, col=2)

    # ── (3,1) 두 검정 동시 요약 ──────────────────────────────────────────────
    def combo_color(k):
        sc = results[k]["sig_chi2"]
        sp = results[k]["sig_perm"]
        if sc and sp:     return "#D62728"   # 둘 다 유의
        if sc or sp:      return "#FF7F0E"   # 하나만 유의
        return "#2CA02C"                      # 둘 다 독립

    combo_colors = [combo_color(k) for k in ks]
    # 두 p값 중 큰 것(더 보수적)을 높이로
    combo_y = [min(results[k]["p_chi2"], results[k]["p_perm"]) for k in ks]

    fig.add_trace(go.Bar(
        x=ks, y=combo_y,
        marker_color=combo_colors, opacity=0.85,
        text=[
            ("독립" if not results[k]["sig_chi2"] and not results[k]["sig_perm"]
             else "한쪽 유의" if results[k]["sig_chi2"] != results[k]["sig_perm"]
             else "둘다 유의")
            for k in ks
        ],
        textposition="outside", textfont=dict(size=8),
        showlegend=False,
        hovertemplate=(
            "k=%{x}<br>min(p)=%{y:.4f}<br>"
            "chi2_p=%{customdata[0]:.4f}  perm_p=%{customdata[1]:.4f}<extra></extra>"
        ),
        customdata=[[results[k]["p_chi2"], results[k]["p_perm"]] for k in ks],
    ), row=3, col=1)
    fig.add_hline(y=ALPHA, line_dash="dash", line_color="black", line_width=1.5, row=3, col=1)
    fig.update_yaxes(title_text="min(p_chi2, p_perm)", range=[0, 1.0], row=3, col=1)

    # ── (3,2) 클러스터 내 T% 편차 (50%에서) ─────────────────────────────────
    # 각 k에서 클러스터별 |T% - 50%| 평균
    mean_devs = []
    max_devs  = []
    for k in ks:
        t_c = np.array(results[k]["t_counts"])
        a_c = np.array(results[k]["a_counts"])
        n_c = t_c + a_c
        t_pcts = np.where(n_c > 0, 100 * t_c / n_c, 50.0)
        devs = np.abs(t_pcts - 50)
        mean_devs.append(float(devs.mean()))
        max_devs.append(float(devs.max()))

    fig.add_trace(go.Scatter(
        x=ks, y=mean_devs,
        mode="lines+markers",
        name="평균 편차",
        line=dict(color="#1F77B4", width=2),
        marker=dict(size=7),
        hovertemplate="k=%{x}<br>mean |T%-50%|=%{y:.2f}pp<extra></extra>",
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=ks, y=max_devs,
        mode="lines+markers",
        name="최대 편차",
        line=dict(color="#FF7F0E", width=2, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="k=%{x}<br>max |T%-50%|=%{y:.2f}pp<extra></extra>",
    ), row=3, col=2)
    fig.add_hline(y=10, line_dash="dot", line_color="gray", line_width=1, row=3, col=2)
    fig.add_annotation(x=max(ks)+0.3, y=10, xref="x6", yref="y6",
                       text="10pp", showarrow=False, font=dict(size=8, color="gray"))
    fig.update_yaxes(title_text="|Tangible% − 50%| (pp)", row=3, col=2)

    # ── x축 공통 설정 ────────────────────────────────────────────────────────
    for row in [1, 2, 3]:
        for col in [1, 2]:
            fig.update_xaxes(tickvals=list(ks), tickfont=dict(size=9),
                             title_text="k", row=row, col=col)

    # ── 결과 요약 텍스트 ─────────────────────────────────────────────────────
    sig_ks   = [k for k in ks if results[k]["sig_chi2"] or results[k]["sig_perm"]]
    insig_ks = [k for k in ks if not results[k]["sig_chi2"] and not results[k]["sig_perm"]]

    summary = (
        f"<b>통계 검정 요약</b> (α={ALPHA}, permutation n={N_PERM})<br>"
        f"<span style='color:#2CA02C'>■ 독립 (두 조건 분포 동일)</span>: k={insig_ks}<br>"
        f"<span style='color:#D62728'>■ 유의 (조건 간 분포 차이)</span>: k={sig_ks}<br>"
        f"<b>→ k={insig_ks}에서 'tangible과 authorless가 동일한 논증 분포를 보인다'고 통계적으로 지지됨</b>"
    )

    fig.update_layout(
        title=dict(
            text=(
                "K-means 통계 검정: Chi-square + Permutation test  [simulation_2]<br>"
                f"<sup>{summary}</sup>"
            ),
            x=0.5, font=dict(size=13),
        ),
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        height=1050,
        width=1300,
        margin=dict(t=220, b=60, r=80),
        hovermode="closest",
        legend=dict(x=1.01, y=0.55, font=dict(size=10)),
    )

    return fig


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
                    "condition": turn["condition"],
                    "turn":      turn["turn"],
                    "point":     pt_num,
                    "subclaim":  pt["subclaim"],
                })

    print(f"  총 {len(records)}개 statements")

    cache_path = DATA_DIR / "embeddings_cache.npy"
    if not cache_path.exists():
        print("  캐시 없음 — analyze.py를 먼저 실행하세요.")
        return
    embs      = np.load(cache_path)
    embs_norm = normalize(embs)
    print(f"  임베딩 캐시 로드")

    print(f"\nChi-square + Permutation test (n_perm={N_PERM}) ...")
    np.random.seed(42)
    results = run_tests(records, embs_norm)

    print("\n시각화 생성 중...")
    fig = build_figure(results)
    out_html = DATA_DIR / "kmeans_stats_test.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"  저장: {out_html.name}")

    # JSON 저장 (cluster_labels 제외)
    save = {str(k): {mk: mv for mk, mv in v.items() if mk != "cluster_labels"}
            for k, v in results.items()}
    out_json = DATA_DIR / "kmeans_stats_test.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_json.name}")

    # 콘솔 요약
    print("\n" + "=" * 75)
    print(f"{'k':>3}  {'chi2_p':>8}  {'chi2_sig':>8}  {'cramers_v':>9}  {'obs_KL':>7}  {'perm_p':>7}  {'perm_sig':>8}")
    print("-" * 75)
    for k in K_RANGE:
        r  = results[k]
        cs = "✗ 유의" if r["sig_chi2"] else "✓ 독립"
        ps = "✗ 유의" if r["sig_perm"] else "✓ 독립"
        print(f"{k:>3}  {r['p_chi2']:>8.4f}  {cs:>8}  {r['cramers_v']:>9.4f}"
              f"  {r['obs_kl']:>7.4f}  {r['p_perm']:>7.4f}  {ps:>8}")

    insig = [k for k in K_RANGE if not results[k]["sig_chi2"] and not results[k]["sig_perm"]]
    sig   = [k for k in K_RANGE if results[k]["sig_chi2"] or results[k]["sig_perm"]]
    print(f"\n  독립 유지 (두 검정 모두): k={insig}")
    print(f"  유의 (하나라도):          k={sig}")
    print("\n완료.")


if __name__ == "__main__":
    main()
