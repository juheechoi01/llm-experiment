#!/usr/bin/env python3
"""
KoBERTopic 분석 (simulations_animal)
  - 데이터: authorless.jsonl / interest.jsonl / neutral.jsonl
  - 임베딩: sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens
  - 토크나이저: MeCab + mecab-ko-dic
  - 나머지 파이프라인: BERTopic (UMAP → HDBSCAN → c-TF-IDF)
  - GPT 토픽 레이블 + 조건별 분포 시각화
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

MECAB_RC    = "-r /opt/homebrew/etc/mecabrc"
KO_MODEL    = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
KEEP_TAGS   = {"NNG", "NNP", "VV", "VA", "XR"}
CONDITIONS  = ["authorless", "interest", "neutral"]
COND_COLORS = {
    "authorless": "#4E79A7",
    "interest":   "#F28E2B",
    "neutral":    "#59A14F",
}

_mecab_tagger = MeCab.Tagger(MECAB_RC)

# 단락형 파싱용 패턴
_TRANS_SPLIT = re.compile(
    r'(?<=[다습요])[.。]?\s+'
    r'(?=(?:또한|마지막으로|더불어|아울러|뿐만\s*아니라|또\s*다른|두\s*번째|세\s*번째|둘째|셋째)[,，]?\s*)'
)
_ELAB_START = re.compile(r'^(?:이[는를로한]\s|이와\s|이러한\s|이로\s|따라서\s|그러므로\s|즉\s|다시\s말해|이를)')
_SENT_SPLIT = re.compile(r'(?<=[다습니까요!?.])\s+')


def mecab_analyzer(text: str) -> list[str]:
    tokens = []
    for line in _mecab_tagger.parse(str(text)).splitlines():
        if line == "EOS" or "\t" not in line:
            continue
        surface, feature = line.split("\t", 1)
        tag = feature.split(",")[0]
        if tag in KEEP_TAGS and len(surface) > 1:
            tokens.append(surface)
    return tokens or ["없음"]


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


def _para_transition(content):
    segs = _TRANS_SPLIT.split(content.strip())
    if len(segs) < 2:
        return {}
    points = {}
    sents0 = [s.strip() for s in _SENT_SPLIT.split(segs[0].strip()) if s.strip()]
    arg1 = sents0[1] if len(sents0) > 1 else sents0[0]
    sub1 = ' '.join(sents0[2:]).strip() if len(sents0) > 2 else ''
    if arg1:
        points[1] = {"statement": arg1.rstrip('.'), "subclaim": sub1}
    for i, seg in enumerate(segs[1:3], 2):
        sents = [s.strip() for s in _SENT_SPLIT.split(seg.strip()) if s.strip()]
        if sents:
            points[i] = {"statement": sents[0].rstrip('.'), "subclaim": ' '.join(sents[1:]).strip()}
    return points


def _para_sentence(content):
    sents = [s.strip() for s in _SENT_SPLIT.split(content.strip())
             if s.strip() and len(s.strip()) > 10]
    arg_sents = [s for s in sents if not _ELAB_START.match(s)]
    if len(arg_sents) > 1:
        arg_sents = arg_sents[1:]
    points = {}
    for i, s in enumerate(arg_sents[:3], 1):
        points[i] = {"statement": s.rstrip('.'), "subclaim": ""}
    return points


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
    if not points:
        points = _para_transition(content)
    if not points:
        points = _para_sentence(content)
    return points


def symmetric_kl(p_arr, q_arr):
    p = np.array(p_arr, dtype=float) + 1e-9
    q = np.array(q_arr, dtype=float) + 1e-9
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


def jensen_shannon_divergence(p_arr, q_arr):
    p = np.array(p_arr, dtype=float) + 1e-12
    q = np.array(q_arr, dtype=float) + 1e-12
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


# ─── 임베딩 (캐시) ───────────────────────────────────────────────────────────

def get_ko_embeddings(texts, cache_path: Path):
    if cache_path.exists():
        print(f"  임베딩 캐시 로드: {cache_path.name}")
        return np.load(cache_path)
    print(f"  xlm-r 임베딩 계산 중 ({len(texts)}개)...")
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
                "아래는 중증 질환 치료 연구를 위한 동물 실험 허용 여부 관련 논증 문장들의 토픽 클러스터입니다.\n"
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

    topic_info: dict[int, dict] = {}
    for tid in sorted(set(topic_ids)):
        mask     = [i for i, t in enumerate(topic_ids) if t == tid]
        recs     = [records[i] for i in mask]
        cond_cnt = {c: sum(1 for r in recs if r["condition"] == c) for c in CONDITIONS}
        cond_pct = {c: round(100 * cond_cnt[c] / len(mask), 1) if mask else 0 for c in CONDITIONS}
        rep_docs = topic_model.get_representative_docs(tid) or [texts[i] for i in mask[:5]]
        topic_info[tid] = {
            "count":               len(mask),
            "condition_counts":    cond_cnt,
            "condition_pcts":      cond_pct,
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

    return topic_model, topic_ids, topic_info, n_topics, n_noise


# ─── 결과 출력 ───────────────────────────────────────────────────────────────

def print_results(records, topic_info, n_topics, n_noise, topic_ids):
    non_noise = {t: v for t, v in topic_info.items() if t != -1}

    cond_totals = {c: sum(1 for r in records if r["condition"] == c) for c in CONDITIONS}
    total       = len(records)

    divergences = {}
    conds = CONDITIONS
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            ca, cb = conds[i], conds[j]
            ca_counts = [non_noise[t]["condition_counts"][ca] for t in sorted(non_noise)]
            cb_counts = [non_noise[t]["condition_counts"][cb] for t in sorted(non_noise)]
            kl  = symmetric_kl(ca_counts, cb_counts)
            jsd = jensen_shannon_divergence(ca_counts, cb_counts)
            divergences[(ca, cb)] = {"kl": kl, "jsd": jsd}

    print(f"\n{'='*80}")
    print(f"KoBERTopic 결과  (noise 제외 {n_topics}개 토픽)")
    print(f"{'='*80}")
    for c in CONDITIONS:
        print(f"  {c}: {cond_totals[c]}개 ({100*cond_totals[c]/total:.0f}%)")
    print()
    header = f"  {'Topic':>6}  {'N':>5}  " + "  ".join(f"{c[:8]:>8}%" for c in CONDITIONS) + "  레이블  /  키워드"
    print(header)
    print(f"  {'-'*6}  {'-'*5}  " + "  ".join(["-"*9] * len(CONDITIONS)) + "  " + "-"*50)
    for tid in sorted(non_noise):
        info  = non_noise[tid]
        label = info.get("label", "")
        kws   = ", ".join(info["keywords"][:4])
        pcts  = "  ".join(f"{info['condition_pcts'][c]:>8.1f}%" for c in CONDITIONS)
        print(f"  {tid:>6}  {info['count']:>5}  {pcts}  {label}  [{kws}]")
    if -1 in topic_info:
        info  = topic_info[-1]
        pcts  = "  ".join(f"{info['condition_pcts'][c]:>8.1f}%" for c in CONDITIONS)
        print(f"  {'noise':>6}  {info['count']:>5}  {pcts}  Noise")

    print(f"\n  ▶ 조건별 분포 발산 (noise 제외):")
    for (ca, cb), divs in divergences.items():
        print(f"     {ca} vs {cb}: KL={divs['kl']:.4f}, JSD={divs['jsd']:.4f}")

    return divergences, cond_totals


# ─── 시각화 ──────────────────────────────────────────────────────────────────

def visualize(topic_model, texts, embs_norm, records, topic_ids, topic_info,
              n_topics, n_noise, divergences, cond_totals):

    non_noise = {t: v for t, v in topic_info.items() if t != -1}

    print("\n시각화 저장 중...")
    try:
        fig = topic_model.visualize_documents(
            texts, embeddings=embs_norm, custom_labels=True,
            title="KoBERTopic — Point Statements [simulations_animal]",
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

    # 조건별 UMAP 2D scatter (3 subplots)
    try:
        print("  조건별 UMAP 2D scatter 계산 중...")
        reducer = umap_lib.UMAP(n_components=2, random_state=42,
                                 min_dist=0.1, n_neighbors=15, metric="cosine")
        coords = reducer.fit_transform(embs_norm)

        all_tids       = sorted(set(topic_ids))
        non_noise_tids = [t for t in all_tids if t != -1]
        color_seq      = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        tid_color      = {tid: color_seq[i % len(color_seq)]
                          for i, tid in enumerate(non_noise_tids)}
        tid_color[-1]  = "#CCCCCC"

        x_all   = coords[:, 0].tolist()
        y_all   = coords[:, 1].tolist()
        x_range = [min(x_all) - 0.5, max(x_all) + 0.5]
        y_range = [min(y_all) - 0.5, max(y_all) + 0.5]

        n_conds  = len(CONDITIONS)
        fig_cond = make_subplots(
            rows=1, cols=n_conds,
            subplot_titles=CONDITIONS,
            horizontal_spacing=0.06,
        )

        for col, cond in enumerate(CONDITIONS, start=1):
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

            fig_cond.add_annotation(
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.99, y=0.01, row=1, col=col,
                text=f"n={len(cond_idx)}",
                showarrow=False, font=dict(size=9, color="#555"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="#ccc", borderwidth=1, borderpad=4,
            )

        div_str = "  |  ".join(
            f"{ca} vs {cb}: KL={d['kl']:.4f} JSD={d['jsd']:.4f}"
            for (ca, cb), d in divergences.items()
        )
        fig_cond.update_layout(
            title=dict(
                text=(f"KoBERTopic — 조건별 문서 분포 [simulations_animal]<br>"
                      f"<sup>총 {len(records)}개  |  토픽 수: {n_topics}  |  "
                      f"noise: {n_noise}개  |  {div_str}<br>"
                      f"임베딩: {KO_MODEL}  |  토크나이저: MeCab+mecab-ko-dic</sup>"),
                x=0.5, font=dict(size=12),
            ),
            plot_bgcolor="#f9f9f9", paper_bgcolor="white",
            legend=dict(title="토픽", font=dict(size=10), x=1.01, y=0.95),
            height=560, width=1700,
            margin=dict(t=130, b=60, r=200),
            hovermode="closest",
        )
        for col in range(1, n_conds + 1):
            fig_cond.update_xaxes(range=x_range, showgrid=False, zeroline=False,
                                   showticklabels=False, title_text="UMAP-1", row=1, col=col)
            fig_cond.update_yaxes(range=y_range, showgrid=False, zeroline=False,
                                   showticklabels=False,
                                   title_text="UMAP-2" if col == 1 else "", row=1, col=col)
        fig_cond.write_html(str(DATA_DIR / "kobertopic_condition_dist.html"), include_plotlyjs="cdn")
        print("  저장: kobertopic_condition_dist.html")
    except Exception as e:
        print(f"  조건별 scatter 생략: {e}")

    # 토픽별 조건 분포 막대 그래프
    try:
        tids    = sorted(non_noise.keys())
        xlabels = [f"T{t}: {non_noise[t].get('label','')[:20]}" for t in tids]

        fig_bar = go.Figure()
        for cond in CONDITIONS:
            pcts = [non_noise[t]["condition_pcts"][cond] for t in tids]
            fig_bar.add_trace(go.Bar(
                name=cond, x=xlabels, y=pcts,
                marker_color=COND_COLORS[cond],
            ))

        div_lines = "  |  ".join(
            f"{ca} vs {cb}: KL={d['kl']:.4f} JSD={d['jsd']:.4f}"
            for (ca, cb), d in divergences.items()
        )
        fig_bar.update_layout(
            barmode="group",
            title=f"KoBERTopic 토픽별 조건 분포 [simulations_animal]<br><sup>{div_lines}</sup>",
            yaxis_title="비율 (%)",
            xaxis_tickangle=-25,
            legend=dict(orientation="h", y=1.15),
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
            "sim_id":         record.get("sim_id", ""),
            "condition":      record["condition"],
            "turn":           record["turn"],
            "point":          record["point"],
            "topic_id":       topic_id,
            "topic_label":    info.get("label", "Noise (미분류)" if topic_id == -1 else f"T{topic_id}"),
            "topic_keywords": ", ".join(info.get("keywords", [])),
            "text":           record["text"],
            "subclaim":       record["subclaim"],
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
    all_sims = []
    for cond in CONDITIONS:
        path = DATA_DIR / f"{cond}.jsonl"
        if path.exists():
            all_sims.extend(load_jsonl(path))
        else:
            print(f"  경고: {path.name} 없음")

    all_turns = get_assistant_turns(all_sims)

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

    texts = [r["text"] for r in records]
    print(f"  총 {len(texts)}개 statements")
    for c in CONDITIONS:
        cnt = sum(1 for r in records if r["condition"] == c)
        print(f"    {c}: {cnt}개")

    embs      = get_ko_embeddings(texts, DATA_DIR / "ko_embeddings_cache.npy")
    embs_norm = normalize(embs)

    topic_model, topic_ids, topic_info, n_topics, n_noise = \
        run_kobertopic(texts, embs_norm, records)

    divergences, cond_totals = print_results(records, topic_info, n_topics, n_noise, topic_ids)

    visualize(topic_model, texts, embs_norm, records,
              topic_ids, topic_info, n_topics, n_noise, divergences, cond_totals)

    save_argument_topic_assignments(records, topic_ids, topic_info)

    output = {
        "model":      KO_MODEL,
        "tokenizer":  "MeCab + mecab-ko-dic",
        "conditions": CONDITIONS,
        "n_topics":   n_topics,
        "n_noise":    n_noise,
        "noise_pct":  round(100 * n_noise / len(topic_ids), 1),
        "divergences": {
            f"{ca}_vs_{cb}": {"kl": round(d["kl"], 4), "jsd": round(d["jsd"], 4)}
            for (ca, cb), d in divergences.items()
        },
        "topics": {str(t): v for t, v in topic_info.items()},
    }
    out_path = DATA_DIR / "kobertopic_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path.name}")
    print("완료.")


if __name__ == "__main__":
    main()
