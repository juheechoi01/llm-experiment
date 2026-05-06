#!/usr/bin/env python3
"""
KoBERTopic 분석 (ukairia777/KoBERTopic 방식)
  - 임베딩: jhgan/ko-sroberta-multitask  (한국어 특화 SentenceTransformer)
  - 토크나이저: MeCab + mecab-ko-dic  (한국어 형태소 분석, 명사/동사/형용사만)
  - 나머지 파이프라인: BERTopic (UMAP → HDBSCAN → c-TF-IDF)
  - GPT 토픽 레이블 + 조건별 분포 시각화

outputs:
  ko_embeddings_cache.npy
  kobertopic_results.json
  kobertopic_barchart.html
  kobertopic_topics.html
  kobertopic_documents.html
  kobertopic_condition_dist.html
  kobertopic_topic_distribution.html
"""

import json
import re
import csv
from pathlib import Path
import MeCab
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
import umap as umap_lib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")
client = OpenAI()

MECAB_RC  = "-r /opt/homebrew/etc/mecabrc"
KO_MODEL  = "jhgan/ko-sroberta-multitask"
KEEP_TAGS = {"NNG", "NNP", "VV", "VA", "XR"}

# 모듈 레벨 tagger (클래스 인스턴스는 sklearn callable analyzer로 동작 불안정)
_mecab_tagger = MeCab.Tagger(MECAB_RC)


def mecab_analyzer(text: str) -> list[str]:
    """CountVectorizer analyzer — 명사·동사·형용사만 추출."""
    tokens = []
    for line in _mecab_tagger.parse(str(text)).splitlines():
        if line == "EOS" or "\t" not in line:
            continue
        surface, feature = line.split("\t", 1)
        tag = feature.split(",")[0]
        if tag in KEEP_TAGS and len(surface) > 1:
            tokens.append(surface)
    return tokens or ["없음"]  # 빈 토큰 방지


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


def jensen_shannon_divergence(t_arr, a_arr):
    t = np.array(t_arr, dtype=float) + 1e-12
    a = np.array(a_arr, dtype=float) + 1e-12
    t /= t.sum(); a /= a.sum()
    m = 0.5 * (t + a)
    return float(0.5 * np.sum(t * np.log2(t / m)) + 0.5 * np.sum(a * np.log2(a / m)))


# ─── 임베딩 (캐시) ───────────────────────────────────────────────────────────

def get_ko_embeddings(texts, cache_path: Path):
    if cache_path.exists():
        print(f"  임베딩 캐시 로드: {cache_path.name}")
        return np.load(cache_path)
    print(f"  ko-sroberta 임베딩 계산 중 ({len(texts)}개)...")
    model = SentenceTransformer(KO_MODEL)
    embs  = model.encode(texts, batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True)
    np.save(cache_path, embs.astype(np.float32))
    print(f"  캐시 저장: {cache_path.name}")
    return embs.astype(np.float32)


# ─── GPT 토픽 레이블 ─────────────────────────────────────────────────────────

def label_topics_gpt(topic_info: dict) -> dict[int, str]:
    prompt_lines = []
    for tid, info in sorted(topic_info.items()):
        if tid == -1:
            continue
        docs = info["representative_docs"][:6]
        kws  = ", ".join(info["keywords"][:6])
        prompt_lines.append(
            f"Topic {tid} [키워드: {kws}]:\n" + "\n".join(f"  - {d}" for d in docs)
        )
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


# ─── BERTopic 학습 ───────────────────────────────────────────────────────────

def run_kobertopic(texts, embs_norm, records):
    print("\nBERTopic 컴포넌트 설정...")

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
        analyzer=mecab_analyzer,
        min_df=1,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=True,
    )

    print("\nKoBERTopic 학습 중...")
    topic_ids, _ = topic_model.fit_transform(texts, embeddings=embs_norm)

    topic_info_df = topic_model.get_topic_info()
    n_topics = len(topic_info_df[topic_info_df["Topic"] != -1])
    n_noise  = sum(1 for t in topic_ids if t == -1)
    print(f"\n  발견된 토픽 수: {n_topics}  (noise: {n_noise}개, {100*n_noise/len(topic_ids):.1f}%)")

    # 토픽별 통계
    topic_info: dict[int, dict] = {}
    for tid in sorted(set(topic_ids)):
        mask  = [i for i, t in enumerate(topic_ids) if t == tid]
        recs  = [records[i] for i in mask]
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

    # GPT 레이블
    print("\nGPT 토픽 레이블링...")
    gpt_labels = label_topics_gpt(topic_info)
    for tid, label in gpt_labels.items():
        if tid in topic_info:
            topic_info[tid]["label"] = label
    if -1 in topic_info:
        topic_info[-1]["label"] = "Noise (미분류)"

    return topic_model, topic_ids, topic_info, n_topics, n_noise


# ─── 결과 출력 ───────────────────────────────────────────────────────────────

def print_results(records, topic_info, n_topics, n_noise, topic_ids):
    non_noise    = {t: v for t, v in topic_info.items() if t != -1}
    expected_t   = sum(1 for r in records if r["condition"] == "tangible") / len(records)
    t_counts     = [non_noise[t]["tangible"]   for t in sorted(non_noise)]
    a_counts     = [non_noise[t]["authorless"] for t in sorted(non_noise)]
    kl           = symmetric_kl(t_counts, a_counts)
    jsd          = jensen_shannon_divergence(t_counts, a_counts)

    print(f"\n{'='*70}")
    print(f"KoBERTopic 결과  (noise 제외 {n_topics}개 토픽)")
    print(f"{'='*70}")
    print(f"기준선: tangible {expected_t*100:.0f}% / authorless {(1-expected_t)*100:.0f}%\n")
    print(f"  {'Topic':>6}  {'N':>5}  {'T%':>6}  {'A%':>6}  레이블  /  키워드")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*50}")
    for tid in sorted(non_noise):
        info  = non_noise[tid]
        label = info.get("label", "")
        kws   = ", ".join(info["keywords"][:4])
        print(f"  {tid:>6}  {info['count']:>5}  {info['tangible_pct']:>5.1f}%  "
              f"{info['authorless_pct']:>5.1f}%  {label}  [{kws}]")
    if -1 in topic_info:
        info = topic_info[-1]
        print(f"  {'noise':>6}  {info['count']:>5}  {info['tangible_pct']:>5.1f}%  "
              f"{info['authorless_pct']:>5.1f}%  Noise")
    print(f"\n  ▶ KL divergence (noise 제외): {kl:.4f}")
    print(f"  ▶ Jensen-Shannon divergence (noise 제외, log2): {jsd:.4f}")
    return kl, jsd, expected_t


# ─── 시각화 ──────────────────────────────────────────────────────────────────

def visualize(topic_model, texts, embs_norm, records, topic_ids, topic_info,
              n_topics, n_noise, kl, jsd, expected_t):

    non_noise = {t: v for t, v in topic_info.items() if t != -1}

    # 1) 기본 BERTopic 시각화
    print("\n시각화 저장 중...")
    try:
        fig = topic_model.visualize_documents(
            texts, embeddings=embs_norm, custom_labels=True,
            title="KoBERTopic — Point Statements [simulation_2]",
        )
        fig.write_html(str(DATA_DIR / "kobertopic_documents.html"), include_plotlyjs="cdn")
        print("  저장: kobertopic_documents.html")
    except Exception as e:
        print(f"  문서 scatter 생략: {e}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=n_topics, n_words=8)
        fig.write_html(str(DATA_DIR / "kobertopic_barchart.html"), include_plotlyjs="cdn")
        print("  저장: kobertopic_barchart.html")
    except Exception as e:
        print(f"  바 차트 생략: {e}")

    try:
        fig = topic_model.visualize_topics()
        fig.write_html(str(DATA_DIR / "kobertopic_topics.html"), include_plotlyjs="cdn")
        print("  저장: kobertopic_topics.html")
    except Exception as e:
        print(f"  토픽 맵 생략: {e}")

    # 2) 조건별 UMAP scatter (토픽 색상, 2패널)
    try:
        print("  조건별 UMAP 2D scatter 계산 중...")
        reducer = umap_lib.UMAP(n_components=2, random_state=42,
                                 min_dist=0.1, n_neighbors=15, metric="cosine")
        coords = reducer.fit_transform(embs_norm)

        all_tids     = sorted(set(topic_ids))
        non_noise_tids = [t for t in all_tids if t != -1]
        color_seq    = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        tid_color    = {tid: color_seq[i % len(color_seq)]
                        for i, tid in enumerate(non_noise_tids)}
        tid_color[-1] = "#CCCCCC"

        x_all   = coords[:, 0].tolist()
        y_all   = coords[:, 1].tolist()
        x_range = [min(x_all) - 0.5, max(x_all) + 0.5]
        y_range = [min(y_all) - 0.5, max(y_all) + 0.5]

        fig_cond = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Authorless", "Tangible"],
            horizontal_spacing=0.08,
        )

        for col, cond in enumerate(["authorless", "tangible"], start=1):
            cond_idx  = [i for i, r in enumerate(records) if r["condition"] == cond]
            other_idx = [i for i, r in enumerate(records) if r["condition"] != cond]

            fig_cond.add_trace(go.Scatter(
                x=[x_all[i] for i in other_idx],
                y=[y_all[i] for i in other_idx],
                mode="markers",
                marker=dict(color="#DDDDDD", size=4, opacity=0.25),
                hoverinfo="skip", showlegend=False, name="",
            ), row=1, col=col)

            for tid in all_tids:
                idx_tid = [i for i in cond_idx if topic_ids[i] == tid]
                if not idx_tid:
                    continue
                label = topic_info[tid].get("label", f"T{tid}")
                hover = [
                    f"<b>{label}</b><br>Turn {records[i]['turn']}  Pt {records[i]['point']}"
                    f"<br>{records[i]['text'][:80]}"
                    for i in idx_tid
                ]
                fig_cond.add_trace(go.Scatter(
                    x=[x_all[i] for i in idx_tid],
                    y=[y_all[i] for i in idx_tid],
                    mode="markers",
                    name=label if tid != -1 else "Noise",
                    legendgroup=f"t{tid}",
                    showlegend=(col == 1),
                    marker=dict(color=tid_color[tid], size=6, opacity=0.8,
                                line=dict(width=0.5, color="white")),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                ), row=1, col=col)

            for tid in non_noise_tids:
                idx_tid = [i for i in cond_idx if topic_ids[i] == tid]
                if not idx_tid:
                    continue
                cx = float(np.mean([x_all[i] for i in idx_tid]))
                cy = float(np.mean([y_all[i] for i in idx_tid]))
                fig_cond.add_annotation(
                    x=cx, y=cy, row=1, col=col,
                    text=f"<b>T{tid}</b>",
                    showarrow=False,
                    font=dict(size=9, color=tid_color[tid]),
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor=tid_color[tid], borderwidth=1, borderpad=2,
                )

            n_cond = len(cond_idx)
            fig_cond.add_annotation(
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.99, y=0.01, row=1, col=col,
                text=f"n={n_cond}",
                showarrow=False, font=dict(size=9, color="#555"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="#ccc", borderwidth=1, borderpad=4,
            )

        fig_cond.update_layout(
            title=dict(
                text=(f"KoBERTopic — 조건별 문서 분포 [simulation_2]<br>"
                      f"<sup>총 {len(records)}개  |  토픽 수: {n_topics}  |  "
                      f"noise: {n_noise}개  |  KL divergence: {kl:.4f}  |  "
                      f"JS divergence: {jsd:.4f}  |  "
                      f"임베딩: {KO_MODEL}  |  토크나이저: MeCab+mecab-ko-dic</sup>"),
                x=0.5, font=dict(size=13),
            ),
            plot_bgcolor="#f9f9f9", paper_bgcolor="white",
            legend=dict(title="토픽", font=dict(size=10), x=1.01, y=0.95),
            height=560, width=1300,
            margin=dict(t=110, b=60, r=200),
            hovermode="closest",
        )
        for col in [1, 2]:
            fig_cond.update_xaxes(range=x_range, showgrid=False, zeroline=False,
                                   showticklabels=False, title_text="UMAP-1", row=1, col=col)
            fig_cond.update_yaxes(range=y_range, showgrid=False, zeroline=False,
                                   showticklabels=False,
                                   title_text="UMAP-2" if col == 1 else "", row=1, col=col)
        fig_cond.write_html(str(DATA_DIR / "kobertopic_condition_dist.html"), include_plotlyjs="cdn")
        print("  저장: kobertopic_condition_dist.html")
    except Exception as e:
        print(f"  조건별 scatter 생략: {e}")

    # 3) 토픽별 조건 비율 바 차트
    try:
        tids    = sorted(non_noise.keys())
        t_pcts  = [non_noise[t]["tangible_pct"]  for t in tids]
        a_pcts  = [non_noise[t]["authorless_pct"] for t in tids]
        xlabels = [f"T{t}: {non_noise[t].get('label','')[:20]}" for t in tids]

        fig_bar = go.Figure(data=[
            go.Bar(name="Tangible",   x=xlabels, y=t_pcts, marker_color="#E15759"),
            go.Bar(name="Authorless", x=xlabels, y=a_pcts, marker_color="#4E79A7"),
        ])
        fig_bar.add_hline(y=expected_t * 100, line_dash="dash", line_color="gray",
                          annotation_text=f"기준선 {expected_t*100:.0f}%")
        fig_bar.update_layout(
            barmode="group",
            title=f"KoBERTopic 토픽별 조건 분포 [simulation_2]  (KL={kl:.4f}, JSD={jsd:.4f})",
            yaxis_title="비율 (%)",
            xaxis_tickangle=-25,
            legend=dict(orientation="h", y=1.1),
        )
        fig_bar.write_html(str(DATA_DIR / "kobertopic_topic_distribution.html"), include_plotlyjs="cdn")
        print("  저장: kobertopic_topic_distribution.html")
    except Exception as e:
        print(f"  분포 차트 생략: {e}")


def save_argument_topic_assignments(records, topic_ids, topic_info):
    rows = []
    for record, topic_id in zip(records, topic_ids):
        info = topic_info.get(topic_id, {})
        rows.append({
            "sim_id": record.get("sim_id", ""),
            "condition": record["condition"],
            "turn": record["turn"],
            "point": record["point"],
            "topic_id": topic_id,
            "topic_label": info.get("label", "Noise (미분류)" if topic_id == -1 else f"T{topic_id}"),
            "topic_keywords": ", ".join(info.get("keywords", [])),
            "text": record["text"],
            "subclaim": record["subclaim"],
        })

    json_path = DATA_DIR / "kobertopic_argument_topics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = DATA_DIR / "kobertopic_argument_topics.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  저장: {json_path.name}")
    print(f"  저장: {csv_path.name}")


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
                    "sim_id":    turn["sim_id"],
                    "text":      pt["statement"],
                    "subclaim":  pt["subclaim"],
                    "condition": turn["condition"],
                    "turn":      turn["turn"],
                    "point":     pt_num,
                })

    texts   = [r["text"] for r in records]
    total_t = sum(1 for r in records if r["condition"] == "tangible")
    total_a = sum(1 for r in records if r["condition"] == "authorless")
    print(f"  총 {len(texts)}개 statements  (tangible {total_t} / authorless {total_a})")

    # 임베딩
    embs      = get_ko_embeddings(texts, DATA_DIR / "ko_embeddings_cache.npy")
    embs_norm = normalize(embs)

    # KoBERTopic 학습
    topic_model, topic_ids, topic_info, n_topics, n_noise = \
        run_kobertopic(texts, embs_norm, records)

    # 결과 출력
    kl, jsd, expected_t = print_results(records, topic_info, n_topics, n_noise, topic_ids)

    # 시각화
    visualize(topic_model, texts, embs_norm, records,
              topic_ids, topic_info, n_topics, n_noise, kl, jsd, expected_t)

    # argument별 토픽/라벨 저장
    save_argument_topic_assignments(records, topic_ids, topic_info)

    # JSON 저장
    output = {
        "model":         KO_MODEL,
        "tokenizer":     "MeCab + mecab-ko-dic",
        "n_topics":      n_topics,
        "n_noise":       n_noise,
        "noise_pct":     round(100 * n_noise / len(topic_ids), 1),
        "kl_divergence": round(kl, 4),
        "js_divergence": round(jsd, 4),
        "topics":        {str(t): v for t, v in topic_info.items()},
    }
    out_path = DATA_DIR / "kobertopic_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path.name}")
    print("완료.")


if __name__ == "__main__":
    main()
