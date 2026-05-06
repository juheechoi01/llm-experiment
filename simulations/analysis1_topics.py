#!/usr/bin/env python3
"""
Analysis 1 재실행: k 정당화 + k=3, k=4 전체 결과
  - k=2~6 실루엣 스코어 비교 (k 선택 근거)
  - k=3, k=4 각각: 클러스터 레이블, 대표 문장, KL divergence
  - 결과 저장: operationalization_k3.json, operationalization_k4.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
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


# ─── 임베딩 ────────────────────────────────────────────────────────────────────

def get_embeddings(texts, batch_size=500, model="text-embedding-3-small"):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(input=texts[i:i + batch_size], model=model)
        all_embs.extend([e.embedding for e in resp.data])
    return np.array(all_embs, dtype=np.float32)


# ─── GPT 클러스터 레이블링 ─────────────────────────────────────────────────────

def label_clusters(cluster_texts: dict, k: int) -> dict[int, str]:
    prompt_lines = []
    for cid in range(k):
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


# ─── 클러스터 중심에 가장 가까운 대표 문장 ────────────────────────────────────

def get_representative(embs_norm, cluster_mask, centroid, n=5):
    """코사인 유사도 기준 centroid에 가장 가까운 n개 인덱스 반환."""
    subset_embs = embs_norm[cluster_mask]
    sims = subset_embs @ centroid  # L2-normalized이므로 dot = cosine sim
    top_local = np.argsort(sims)[::-1][:n]
    global_indices = np.where(cluster_mask)[0]
    return global_indices[top_local].tolist()


# ─── KL divergence ────────────────────────────────────────────────────────────

def symmetric_kl(t_counts, a_counts):
    t = np.array(t_counts, dtype=float) + 1e-9
    a = np.array(a_counts, dtype=float) + 1e-9
    t /= t.sum(); a /= a.sum()
    return float(np.sum(t * np.log(t / a)) + np.sum(a * np.log(a / t)))


# ─── 단일 k 분석 ──────────────────────────────────────────────────────────────

def analyze_k(records, embs_norm, k, sil_score, labels_map):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(embs_norm)
    centroids = normalize(km.cluster_centers_)

    cluster_records = defaultdict(list)
    for i, rec in enumerate(records):
        cluster_records[int(cluster_ids[i])].append((i, rec))

    cluster_texts = {cid: [r["text"] for _, r in items]
                     for cid, items in cluster_records.items()}

    print(f"\n  클러스터 레이블링 (k={k})...")
    labels = label_clusters(cluster_texts, k)

    clusters_out = {}
    t_counts, a_counts = [], []

    for cid in range(k):
        items = cluster_records[cid]
        indices = [i for i, _ in items]
        recs    = [r for _, r in items]

        t_cnt = sum(1 for r in recs if r["condition"] == "tangible")
        a_cnt = sum(1 for r in recs if r["condition"] == "authorless")
        total = t_cnt + a_cnt
        t_counts.append(t_cnt)
        a_counts.append(a_cnt)

        # 대표 문장 (centroid 기준 상위 5개)
        mask = (cluster_ids == cid)
        rep_indices = get_representative(embs_norm, mask, centroids[cid], n=5)
        rep_stmts = [records[i]["text"] for i in rep_indices]

        clusters_out[str(cid)] = {
            "label": labels.get(cid, f"Cluster {cid}"),
            "total": total,
            "tangible": t_cnt,
            "authorless": a_cnt,
            "tangible_pct": round(100 * t_cnt / total, 1) if total else 0,
            "authorless_pct": round(100 * a_cnt / total, 1) if total else 0,
            "representative_statements": rep_stmts,
        }

    kl = symmetric_kl(t_counts, a_counts)

    return {
        "k": k,
        "silhouette_score": round(float(sil_score), 4),
        "kl_divergence": round(kl, 4),
        "clusters": clusters_out,
    }


# ─── 메인 ─────────────────────────────────────────────────────────────────────

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
                })

    print(f"  총 {len(records)}개 statement")

    print("  임베딩 계산 중...")
    embs      = get_embeddings([r["text"] for r in records])
    embs_norm = normalize(embs)

    # ── k=2~6 실루엣 스코어 비교 ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("k 선택 근거: 실루엣 스코어 (k = 2 ~ 6)")
    print("=" * 55)

    k_range = [2, 3, 4, 5, 6]
    sil_scores = {}
    km_results = {}   # k → (cluster_ids, km_model)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        ids = km.fit_predict(embs_norm)
        sil = silhouette_score(embs_norm, ids, metric="cosine", sample_size=1000, random_state=42)
        sil_scores[k] = float(sil)
        km_results[k] = (ids, km)

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\n  {'k':>4}  {'Silhouette Score':>18}  {'비고':>15}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*15}")
    for k in k_range:
        note = "← 최고" if k == best_k else ("← 검토 대상" if k in [3, 4] else "")
        bar = "█" * int(sil_scores[k] * 40)
        print(f"  {k:>4}  {sil_scores[k]:>8.4f}  {bar:<20}  {note}")

    print(f"\n  ▶ 실루엣 스코어 최고: k={best_k} ({sil_scores[best_k]:.4f})")

    # ── k=3, k=4 전체 분석 + JSON 저장 ────────────────────────────────────────
    for target_k in [3, 4]:
        print(f"\n{'='*55}")
        print(f"k={target_k} 상세 분석")
        print("=" * 55)

        # k=target_k로 재클러스터링 (get_representative 호환 위해 analyze_k에서 내부 처리)
        result = analyze_k(records, embs_norm, target_k, sil_scores[target_k], {})

        # 결과 출력
        print(f"\n  실루엣 스코어 : {result['silhouette_score']}")
        print(f"  KL divergence : {result['kl_divergence']}  (0에 가까울수록 두 조건 분포 동일)")
        print(f"\n  {'Cluster':>8}  {'N':>5}  {'T%':>6}  {'A%':>6}  레이블")
        print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*30}")
        for cid, cs in result["clusters"].items():
            print(f"  {cid:>8}  {cs['total']:>5}  {cs['tangible_pct']:>5.1f}%  {cs['authorless_pct']:>5.1f}%  {cs['label']}")
            for stmt in cs["representative_statements"][:3]:
                print(f"            · {stmt[:75]}")

        # 저장
        out_path = DATA_DIR / f"operationalization_k{target_k}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n  저장: {out_path}")

    # 실루엣 스코어 요약도 JSON에 추가
    sil_path = DATA_DIR / "silhouette_scores.json"
    with open(sil_path, "w", encoding="utf-8") as f:
        json.dump({"silhouette_by_k": {str(k): round(v, 4) for k, v in sil_scores.items()},
                   "best_k": best_k}, f, indent=2)
    print(f"\n실루엣 스코어 요약 저장: {sil_path}")
    print("\n분석 완료.")


if __name__ == "__main__":
    main()
