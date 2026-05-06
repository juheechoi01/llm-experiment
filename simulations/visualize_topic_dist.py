#!/usr/bin/env python3
"""
BERTopic 조건별 토픽 분포 비교 시각화
  - 각 조건(tangible/authorless) 내에서 토픽이 차지하는 비율 비교
  - 결과: bertopic_topic_distribution.html
"""

import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent

# ─── 데이터 로딩 ───────────────────────────────────────────────────────────────

with open(DATA_DIR / "bertopic_results.json", encoding="utf-8") as f:
    results = json.load(f)

topics = {int(k): v for k, v in results["topics"].items()}
non_noise = {k: v for k, v in topics.items() if k != -1}
kl = results["kl_divergence"]

# 전체 tangible/authorless 문서 수 계산 (noise 포함)
total_tangible   = sum(v["tangible"]   for v in topics.values())
total_authorless = sum(v["authorless"] for v in topics.values())

# 토픽 정렬: tangible 비율 기준 내림차순
tids = sorted(non_noise.keys(), key=lambda t: non_noise[t]["tangible"] / max(non_noise[t]["count"], 1), reverse=True)

labels      = [f"T{t}: {non_noise[t]['label'][:18]}" for t in tids]
counts      = [non_noise[t]["count"]      for t in tids]

# 각 조건 내 토픽 비율 (해당 조건 전체 문서 대비 %)
t_within    = [100 * non_noise[t]["tangible"]   / total_tangible   for t in tids]
a_within    = [100 * non_noise[t]["authorless"] / total_authorless for t in tids]

# 각 토픽 내 조건 비율 (해당 토픽 내 tangible/authorless 비율 %)
t_in_topic  = [non_noise[t]["tangible_pct"]   for t in tids]
a_in_topic  = [non_noise[t]["authorless_pct"] for t in tids]

# 편차: tangible 내 비율 - authorless 내 비율
deviation   = [t - a for t, a in zip(t_within, a_within)]

TANGIBLE_COLOR   = "#E15759"
AUTHORLESS_COLOR = "#4E79A7"
NEUTRAL_COLOR    = "#76B7B2"

# ─── 3-panel 레이아웃 ──────────────────────────────────────────────────────────

fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.42, 0.30, 0.28],
    subplot_titles=[
        "① 각 조건 내 토픽 분포 (조건별 전체 문서 대비 %)",
        "② 각 토픽 내 조건 비율 (토픽 내 T% vs A%)",
        "③ 편차: Tangible% − Authorless% (조건 내 분포 기준)",
    ],
    vertical_spacing=0.10,
)

# ── Panel 1: within-condition distribution ─────────────────────────────────────
fig.add_trace(go.Bar(
    name="Tangible",
    x=labels, y=t_within,
    marker_color=TANGIBLE_COLOR,
    opacity=0.85,
    hovertemplate="<b>%{x}</b><br>Tangible 내 비율: %{y:.1f}%<extra></extra>",
), row=1, col=1)

fig.add_trace(go.Bar(
    name="Authorless",
    x=labels, y=a_within,
    marker_color=AUTHORLESS_COLOR,
    opacity=0.85,
    hovertemplate="<b>%{x}</b><br>Authorless 내 비율: %{y:.1f}%<extra></extra>",
), row=1, col=1)

# 기준선: 만약 두 조건 분포가 완전히 동일하다면 각 토픽의 비율이 같아야 함
fig.add_annotation(
    xref="x domain", yref="y domain",
    x=0.99, y=0.95, row=1, col=1,
    text=f"Symmetric KL divergence = {kl:.4f}",
    showarrow=False, font=dict(size=11),
    bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1, borderpad=4,
    align="right",
)

# ── Panel 2: within-topic condition split ──────────────────────────────────────
fig.add_trace(go.Bar(
    name="Tangible (토픽 내)",
    x=labels, y=t_in_topic,
    marker_color=TANGIBLE_COLOR,
    opacity=0.85,
    showlegend=False,
    hovertemplate="<b>%{x}</b><br>해당 토픽 내 Tangible: %{y:.1f}%<extra></extra>",
), row=2, col=1)

fig.add_trace(go.Bar(
    name="Authorless (토픽 내)",
    x=labels, y=a_in_topic,
    marker_color=AUTHORLESS_COLOR,
    opacity=0.85,
    showlegend=False,
    hovertemplate="<b>%{x}</b><br>해당 토픽 내 Authorless: %{y:.1f}%<extra></extra>",
), row=2, col=1)

# 50% 기준선
fig.add_hline(y=50, line_dash="dash", line_color="gray", line_width=1.2, row=2, col=1,
              annotation_text="50% 기준", annotation_position="top right",
              annotation_font_size=10)

# ── Panel 3: deviation bar (diverging) ────────────────────────────────────────
bar_colors = [TANGIBLE_COLOR if d > 0 else AUTHORLESS_COLOR for d in deviation]
fig.add_trace(go.Bar(
    name="편차 (T% − A%)",
    x=labels, y=deviation,
    marker_color=bar_colors,
    opacity=0.80,
    showlegend=False,
    hovertemplate=(
        "<b>%{x}</b><br>"
        "편차: %{y:+.1f}%p<br>"
        "<i>양수 = Tangible이 더 많이 사용</i><extra></extra>"
    ),
), row=3, col=1)

fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)

# 편차 임계선 ±10%p
fig.add_hrect(y0=-5, y1=5, fillcolor="lightgreen", opacity=0.15,
              line_width=0, row=3, col=1,
              annotation_text="±5%p 이내 (균형)", annotation_position="top right",
              annotation_font_size=9)

# ─── 레이아웃 ──────────────────────────────────────────────────────────────────

fig.update_layout(
    title=dict(
        text=(
            "BERTopic 토픽별 조건 분포 비교 (Tangible vs Authorless)<br>"
            f"<sup>21개 토픽 | noise 제외 | KL divergence = {kl:.4f} | "
            f"tangible {total_tangible}개 / authorless {total_authorless}개</sup>"
        ),
        x=0.5, font=dict(size=15),
    ),
    barmode="group",
    plot_bgcolor="#f9f9f9",
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    height=950,
    width=1150,
    margin=dict(t=110, b=60),
    hovermode="x unified",
)

# y축 레이블
fig.update_yaxes(title_text="조건 내 비율 (%)", row=1, col=1)
fig.update_yaxes(title_text="토픽 내 비율 (%)", row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="편차 (%p)", row=3, col=1)

# x축 기울기
for row in [1, 2, 3]:
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9), row=row, col=1)

# ─── 저장 & 열기 ───────────────────────────────────────────────────────────────

out_path = DATA_DIR / "bertopic_topic_distribution.html"
fig.write_html(str(out_path), include_plotlyjs="cdn")
print(f"저장 완료: {out_path}")
