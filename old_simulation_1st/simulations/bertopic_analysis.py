#!/usr/bin/env python3
"""
BERTopic 토픽 모델링
  - OpenAI 임베딩 → UMAP(5D) → HDBSCAN → c-TF-IDF → GPT 토픽 레이블
  - 조건별(tangible/authorless) 토픽 분포 + KL divergence
  - 임베딩은 embeddings_cache.npy에 캐시 (재실행 시 API 호출 생략)
  - 결과: bertopic_results.json, bertopic_*.html
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import normalize
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")
client = OpenAI()


# ─── 데이터 로딩 & 파싱 ────────────────────────────────────────────────────────

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
                newline_split = re.split(r'\n\s+', text, maxsplit=1)
                if len(newline_split) == 2:
                    stmt = newline_split[0].strip().rstrip('.')
                    sub  = newline_split[1].strip()
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


# ─── 임베딩 (캐시 지원) ────────────────────────────────────────────────────────

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


# ─── GPT 토픽 레이블링 ─────────────────────────────────────────────────────────

def label_topics_gpt(topic_info: dict) -> dict[int, str]:
    """각 토픽의 대표 문장 5개를 GPT에 넘겨 한국어 레이블 생성."""
    prompt_lines = []
    for tid, info in topic_info.items():
        if tid == -1:
            continue
        docs = info["representative_docs"][:6]
        prompt_lines.append(f"Topic {tid}:\n" + "\n".join(f"  - {d}" for d in docs))

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


# ─── KL divergence ─────────────────────────────────────────────────────────────

def symmetric_kl(t_counts, a_counts):
    t = np.array(t_counts, dtype=float) + 1e-9
    a = np.array(a_counts, dtype=float) + 1e-9
    t /= t.sum(); a /= a.sum()
    return float(np.sum(t * np.log(t / a)) + np.sum(a * np.log(a / t)))


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    # 데이터 로딩
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
    print(f"  총 {len(texts)}개 statement  (tangible {sum(1 for r in records if r['condition']=='tangible')}, "
          f"authorless {sum(1 for r in records if r['condition']=='authorless')})")

    # 임베딩 (캐시)
    embs      = get_embeddings_cached(texts, DATA_DIR / "embeddings_cache.npy")
    embs_norm = normalize(embs)

    # ── BERTopic 컴포넌트 설정 ────────────────────────────────────────────────
    umap_model = UMAP(
        n_components=5,      # 클러스터링용 5D
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=25,
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

    # ── 학습 ────────────────────────────────────────────────────────────────
    print("\nBERTopic 학습 중...")
    topic_ids, _ = topic_model.fit_transform(texts, embeddings=embs_norm)

    topic_info_df = topic_model.get_topic_info()
    n_topics = len(topic_info_df[topic_info_df["Topic"] != -1])
    n_noise  = sum(1 for t in topic_ids if t == -1)
    print(f"\n  발견된 토픽 수: {n_topics}  (noise=-1: {n_noise}개, {100*n_noise/len(topic_ids):.1f}%)")

    # 각 토픽의 대표 문장 수집
    topic_info: dict[int, dict] = {}
    for tid in sorted(set(topic_ids)):
        mask  = [i for i, t in enumerate(topic_ids) if t == tid]
        recs  = [records[i] for i in mask]
        t_cnt = sum(1 for r in recs if r["condition"] == "tangible")
        a_cnt = sum(1 for r in recs if r["condition"] == "authorless")

        # BERTopic 대표 문장 (내부 저장된 것)
        rep_docs = topic_model.get_representative_docs(tid) or [texts[i] for i in mask[:5]]

        topic_info[tid] = {
            "count":             len(mask),
            "tangible":          t_cnt,
            "authorless":        a_cnt,
            "tangible_pct":      round(100 * t_cnt / len(mask), 1) if mask else 0,
            "authorless_pct":    round(100 * a_cnt / len(mask), 1) if mask else 0,
            "representative_docs": rep_docs,
            "keywords":          [w for w, _ in (topic_model.get_topic(tid) or [])[:8]],
        }

    # ── GPT 레이블 ──────────────────────────────────────────────────────────
    print("\nGPT 토픽 레이블링...")
    gpt_labels = label_topics_gpt(topic_info)
    for tid, label in gpt_labels.items():
        if tid in topic_info:
            topic_info[tid]["label"] = label
    if -1 in topic_info:
        topic_info[-1]["label"] = "Noise (미분류)"

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    expected_t = sum(1 for r in records if r["condition"] == "tangible") / len(records)

    print(f"\n{'='*70}")
    print(f"BERTopic 결과 (noise 제외 {n_topics}개 토픽)")
    print(f"{'='*70}")
    print(f"기준선: tangible {expected_t*100:.0f}% / authorless {(1-expected_t)*100:.0f}%\n")
    print(f"  {'Topic':>6}  {'N':>5}  {'T%':>6}  {'A%':>6}  키워드 / 레이블")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*40}")

    non_noise = {t: v for t, v in topic_info.items() if t != -1}
    for tid in sorted(non_noise):
        info  = non_noise[tid]
        label = info.get("label", "")
        kws   = ", ".join(info["keywords"][:4])
        print(f"  {tid:>6}  {info['count']:>5}  {info['tangible_pct']:>5.1f}%  {info['authorless_pct']:>5.1f}%  [{kws}]  {label}")

    if -1 in topic_info:
        info = topic_info[-1]
        print(f"  {'noise':>6}  {info['count']:>5}  {info['tangible_pct']:>5.1f}%  {info['authorless_pct']:>5.1f}%  Noise")

    # KL divergence (noise 제외)
    t_counts = [non_noise[t]["tangible"]  for t in sorted(non_noise)]
    a_counts = [non_noise[t]["authorless"] for t in sorted(non_noise)]
    kl = symmetric_kl(t_counts, a_counts)
    print(f"\n  ▶ KL divergence (noise 제외): {kl:.4f}")
    print(f"     (0에 가까울수록 두 조건이 같은 토픽 분포)")

    # ── 시각화 저장 ──────────────────────────────────────────────────────────
    print("\n시각화 저장 중...")

    # 1) 문서 UMAP scatter (조건별 색상)
    conditions = [r["condition"] for r in records]
    try:
        fig_docs = topic_model.visualize_documents(
            texts,
            embeddings=embs_norm,
            custom_labels=True,
            title="BERTopic — Point Statements (tangible ● / authorless ◆)",
        )
        fig_docs.write_html(str(DATA_DIR / "bertopic_documents.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_documents.html")
    except Exception as e:
        print(f"  문서 시각화 생략: {e}")

    # 2) 토픽 바 차트 (키워드)
    try:
        fig_bar = topic_model.visualize_barchart(top_n_topics=n_topics, n_words=8)
        fig_bar.write_html(str(DATA_DIR / "bertopic_barchart.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_barchart.html")
    except Exception as e:
        print(f"  바 차트 생략: {e}")

    # 3) 인터토픽 거리 맵
    try:
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html(str(DATA_DIR / "bertopic_topics.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_topics.html")
    except Exception as e:
        print(f"  토픽 맵 생략: {e}")

    # 4) 조건별 토픽 분포 (tangible vs authorless) — 직접 제작
    try:
        import plotly.graph_objects as go

        tids    = sorted(non_noise.keys())
        t_pcts  = [non_noise[t]["tangible_pct"]   for t in tids]
        a_pcts  = [non_noise[t]["authorless_pct"]  for t in tids]
        labels  = [non_noise[t].get("label", f"T{t}")[:20] for t in tids]

        fig_cond = go.Figure(data=[
            go.Bar(name="Tangible",   x=labels, y=t_pcts, marker_color="#E15759"),
            go.Bar(name="Authorless", x=labels, y=a_pcts, marker_color="#4E79A7"),
        ])
        fig_cond.add_hline(y=expected_t * 100, line_dash="dash", line_color="gray",
                           annotation_text=f"기준선 {expected_t*100:.0f}%")
        fig_cond.update_layout(
            barmode="group",
            title="토픽별 조건 분포 (Tangible vs Authorless)",
            yaxis_title="비율 (%)",
            xaxis_title="토픽",
            legend=dict(orientation="h", y=1.1),
        )
        fig_cond.write_html(str(DATA_DIR / "bertopic_condition_dist.html"), include_plotlyjs="cdn")
        print("  저장: bertopic_condition_dist.html")
    except Exception as e:
        print(f"  조건 분포 차트 생략: {e}")

    # ── JSON 저장 ────────────────────────────────────────────────────────────
    output = {
        "n_topics":      n_topics,
        "n_noise":       n_noise,
        "noise_pct":     round(100 * n_noise / len(topic_ids), 1),
        "kl_divergence": round(kl, 4),
        "topics":        {str(t): v for t, v in topic_info.items()},
    }
    out_path = DATA_DIR / "bertopic_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
    print("분석 완료.")


if __name__ == "__main__":
    main()
