#!/usr/bin/env python3
"""
Experiment Operationalization Check: authorless vs tangible

1. Topic modeling on pooled point statements (tangible + authorless)
   → 두 조건이 같은 논증 공간을 쓰는지 확인
2. Sub-claim org mention rate: tangible cites orgs, authorless doesn't
3. Extract all orgs cited in tangible and verify whether they're real
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from openai import OpenAI
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
load_dotenv(DATA_DIR.parent / ".env")
client = OpenAI()


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_assistant_turns(simulations):
    turns = []
    for sim in simulations:
        for t in sim["conversation"]:
            if t["role"] == "assistant":
                turns.append({
                    "sim_id": sim["sim_id"],
                    "condition": sim["condition"],
                    "turn": t["turn"],
                    "content": t["content"],
                })
    return turns


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_response(content):
    """
    Parse assistant response into:
      - main_claim: opening sentence before numbered points
      - points: {1: {"statement": ..., "subclaim": ..., "full": ...}, ...}
    """
    content = content.strip()

    # Extract main claim: text before first numbered point (or first bullet)
    main_claim_match = re.match(r'^(.+?)(?=\n\s*\d+\.|\n\s*[-•])', content, re.DOTALL)
    main_claim = main_claim_match.group(1).strip() if main_claim_match else ""

    points = {}

    # Try numbered format: "1. ..."
    numbered_items = re.findall(
        r'(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\Z)',
        content, re.DOTALL
    )
    if numbered_items:
        for num_str, text in numbered_items:
            num = int(num_str)
            if 1 <= num <= 3:
                _parse_point(num, text.strip(), points)

    # Fallback: bullet format "- ..."
    if not points:
        bullet_items = re.findall(
            r'^[-•]\s+(.+?)(?=\n[-•]|\Z)',
            content, re.DOTALL | re.MULTILINE
        )
        for i, text in enumerate(bullet_items[:3], 1):
            _parse_point(i, text.strip(), points)

    return {"main_claim": main_claim, "points": points}


def _parse_point(num, text, points_dict):
    """Split a numbered/bulleted item into statement (first sentence) and sub-claim (rest)."""
    # First sentence ends at ". " or ".\n" or end of line
    # Handle inline format: "Statement. Sub-claim text" vs newline format "Statement\n  Sub-claim"
    newline_split = re.split(r'\n\s+', text, maxsplit=1)
    if len(newline_split) == 2:
        # Newline-separated: "Statement\n   Sub-claim"
        statement = newline_split[0].strip().rstrip('.')
        subclaim = newline_split[1].strip()
    else:
        # Inline: "Statement. Sub-claim text with org..."
        # Split on first period followed by space+capital or newline
        sentence_split = re.split(r'\.\s+(?=[가-힣A-Z\d])', text, maxsplit=1)
        if len(sentence_split) == 2:
            statement = sentence_split[0].strip()
            subclaim = sentence_split[1].strip()
        else:
            statement = text.strip().rstrip('.')
            subclaim = ""

    points_dict[num] = {
        "statement": statement,
        "subclaim": subclaim,
        "full": text,
    }


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embeddings(texts, model="text-embedding-3-small", batch_size=500):
    """Embed texts in batches (API limit = 2048 inputs per call)."""
    if not texts:
        return []
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embs.extend([np.array(e.embedding) for e in response.data])
    return all_embs


# ─── Analysis 1: Topic Modeling on Pooled Point Statements ───────────────────

def label_clusters(clusters: dict[int, list[str]], k: int) -> dict[int, str]:
    """Ask GPT-4o-mini to name each cluster given up to 8 example statements."""
    cluster_examples = {}
    for cid in range(k):
        examples = clusters[cid][:8]
        cluster_examples[cid] = examples

    prompt_lines = []
    for cid, examples in cluster_examples.items():
        ex_str = "\n".join(f"  - {e}" for e in examples)
        prompt_lines.append(f"Cluster {cid}:\n{ex_str}")

    response = client.chat.completions.create(
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
    raw = json.loads(response.choices[0].message.content).get("labels", {})
    return {int(k): v for k, v in raw.items()}


def analyze_claim_topics(all_turns, k: int = 6):
    print("\n" + "=" * 65)
    print("ANALYSIS 1: Point Statement Topic Modeling (K-means, k={})".format(k))
    print("=" * 65)
    print("목적: 두 조건이 같은 논증 공간을 쓰는지 — point 1/2/3 구분 없이 전체 pooling\n")

    # Collect all point statements with condition label
    records = []  # {"text": ..., "condition": ..., "turn": ..., "point": ...}
    for turn in all_turns:
        parsed = parse_response(turn["content"])
        for pt_num, pt in parsed["points"].items():
            if pt["statement"].strip():
                records.append({
                    "text": pt["statement"],
                    "condition": turn["condition"],
                    "turn": turn["turn"],
                    "point": pt_num,
                })

    print(f"  총 point statements: {len(records)}개")
    t_n = sum(1 for r in records if r["condition"] == "tangible")
    a_n = sum(1 for r in records if r["condition"] == "authorless")
    print(f"  tangible: {t_n}개,  authorless: {a_n}개\n")

    # Embed all statements
    print(f"  임베딩 계산 중...")
    texts = [r["text"] for r in records]
    embs = get_embeddings(texts)
    emb_matrix = normalize(np.array(embs))  # L2-normalize for cosine K-means

    # K-means clustering
    print(f"  K-means 클러스터링 (k={k})...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_matrix)
    for i, rec in enumerate(records):
        rec["cluster"] = int(labels[i])

    # Collect examples per cluster
    cluster_texts: dict[int, list[str]] = defaultdict(list)
    for rec in records:
        cluster_texts[rec["cluster"]].append(rec["text"])

    # GPT label each cluster
    print(f"  클러스터 레이블링 중...")
    cluster_labels = label_clusters(cluster_texts, k)

    # Compute per-cluster distribution (tangible vs authorless)
    cluster_stats: dict[int, dict] = {}
    for cid in range(k):
        members = [r for r in records if r["cluster"] == cid]
        t_cnt = sum(1 for r in members if r["condition"] == "tangible")
        a_cnt = sum(1 for r in members if r["condition"] == "authorless")
        total = t_cnt + a_cnt
        cluster_stats[cid] = {
            "label": cluster_labels.get(cid, f"Cluster {cid}"),
            "total": total,
            "tangible": t_cnt,
            "authorless": a_cnt,
            "tangible_pct": 100 * t_cnt / total if total else 0,
            "authorless_pct": 100 * a_cnt / total if total else 0,
        }

    # Expected split based on overall corpus ratio
    expected_t_pct = 100 * t_n / (t_n + a_n)
    expected_a_pct = 100 * a_n / (t_n + a_n)

    # Print results
    print(f"\n  전체 비율 기준선: tangible {expected_t_pct:.0f}% / authorless {expected_a_pct:.0f}%")
    print(f"  → 각 클러스터가 기준선과 비슷하면 두 조건이 같은 논증을 사용하는 것\n")

    print(f"  {'Cluster':>8}  {'N':>5}  {'Tangible%':>10}  {'Authorless%':>12}  레이블")
    print(f"  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*30}")
    for cid in range(k):
        s = cluster_stats[cid]
        flag = ""
        # Flag if one condition dominates by >20pp over baseline
        if abs(s["tangible_pct"] - expected_t_pct) > 20:
            flag = " ⚠"
        print(f"  {cid:>8}  {s['total']:>5}  {s['tangible_pct']:>9.1f}%  {s['authorless_pct']:>11.1f}%  {s['label']}{flag}")

    # Print top-3 example statements per cluster
    print(f"\n  [클러스터별 대표 문장 (각 3개씩)]")
    for cid in range(k):
        s = cluster_stats[cid]
        print(f"\n  Cluster {cid} — {s['label']}  (T:{s['tangible']} / A:{s['authorless']})")
        for ex in cluster_texts[cid][:3]:
            print(f"    · {ex[:80]}")

    # Summary: KL divergence as imbalance measure
    t_dist = np.array([cluster_stats[c]["tangible"] for c in range(k)], dtype=float)
    a_dist = np.array([cluster_stats[c]["authorless"] for c in range(k)], dtype=float)
    t_dist /= t_dist.sum()
    a_dist /= a_dist.sum()
    # Symmetric KL
    eps = 1e-9
    kl = float(np.sum(t_dist * np.log((t_dist + eps) / (a_dist + eps))) +
               np.sum(a_dist * np.log((a_dist + eps) / (t_dist + eps))))
    print(f"\n  ▶ 조건 간 클러스터 분포 Symmetric KL divergence: {kl:.3f}")
    print(f"     (0에 가까울수록 두 조건이 동일한 논증 분포 → operationalization 성공)")

    return {"cluster_stats": cluster_stats, "kl_divergence": kl}


# ─── Analysis 2: Org Mention in Sub-claims ────────────────────────────────────

# Org detection patterns based on observed tangible data
ORG_PATTERNS = [
    # Korean name (English name in parens)
    re.compile(r'[가-힣A-Za-z\s]{2,}\([A-Z][A-Za-z\s&,\.\/]{5,}\)', re.UNICODE),
    # Known abbreviations
    re.compile(r'\b(FDA|WHO|PhRMA|BIO|IFPMA|EFPIA|NIH|CDC|EMA|FIP|IABS|ILSR|ACS|AdvaMed)\b'),
    # Korean org suffixes
    re.compile(r'[가-힣A-Za-z\s]{3,}(?:협회|기구|연구소|재단|연합|학회|협의회|연맹|연구재단|연구원|싱크탱크|식품의약국)', re.UNICODE),
    # Named pharma/biotech companies
    re.compile(r'\b(Novartis|Pfizer|Amgen|AstraZeneca|노바티스|화이자|암젠|아스트라제네카|Roche|로슈)\b'),
]


def has_org_mention(text):
    return any(p.search(text) for p in ORG_PATTERNS)


def extract_org_names(text):
    orgs = set()
    # Korean (English) pattern — most informative
    for m in re.finditer(r'([가-힣A-Za-z\s]+)\(([A-Za-z][A-Za-z\s&,\.\/]+)\)', text):
        korean = m.group(1).strip()
        english = m.group(2).strip()
        orgs.add(f"{korean} ({english})")
    # Korean org names alone
    for m in re.finditer(r'([가-힣A-Za-z\s]{3,}(?:협회|기구|연구소|재단|연합|학회|협의회|연맹|싱크탱크|식품의약국))', text, re.UNICODE):
        name = m.group(1).strip()
        if not re.search(re.escape(name) + r'\s*\(', text):
            orgs.add(name)
    # Abbreviations
    for abbr in re.findall(r'\b(FDA|WHO|PhRMA|BIO|IFPMA|EFPIA|NIH|CDC|EMA|FIP|IABS|ILSR|ACS|AdvaMed)\b', text):
        orgs.add(abbr)
    # Named companies
    for co in re.findall(r'\b(Novartis|Pfizer|Amgen|AstraZeneca|노바티스|화이자|암젠|로슈|Roche)\b', text):
        orgs.add(co)
    return orgs


def analyze_org_mentions(all_turns):
    print("\n" + "=" * 65)
    print("ANALYSIS 2: Sub-claim 내 기관/조직 언급 비율")
    print("=" * 65)
    print("목적: tangible은 기관을 인용, authorless는 인용하지 않는지 확인\n")

    total = defaultdict(int)
    with_org = defaultdict(int)
    by_turn = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [total, with_org]

    for turn in all_turns:
        cond = turn["condition"]
        parsed = parse_response(turn["content"])
        for pt_num, pt in parsed["points"].items():
            check_text = pt["subclaim"] if pt["subclaim"] else pt["full"]
            total[cond] += 1
            by_turn[cond][turn["turn"]][0] += 1
            if has_org_mention(check_text):
                with_org[cond] += 1
                by_turn[cond][turn["turn"]][1] += 1

    # Overall
    for cond in ["tangible", "authorless"]:
        pct = 100 * with_org[cond] / total[cond] if total[cond] else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {cond.upper()}:  [{bar}] {pct:.1f}%")
        print(f"    기관 언급 sub-claim: {with_org[cond]} / {total[cond]}")

    # Per-turn breakdown
    print(f"\n  Turn별 기관 언급 비율:")
    print(f"  {'Turn':>6}  {'Tangible':>10}  {'Authorless':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
    for t in [1, 2, 3]:
        t_tot, t_org = by_turn["tangible"][t]
        a_tot, a_org = by_turn["authorless"][t]
        t_pct = 100 * t_org / t_tot if t_tot else 0
        a_pct = 100 * a_org / a_tot if a_tot else 0
        print(f"  {t:>6}  {t_pct:>9.1f}%  {a_pct:>9.1f}%")

    return {"total": dict(total), "with_org": dict(with_org)}


# ─── Analysis 3: Extract & Verify Organizations ───────────────────────────────

def analyze_org_verification(all_turns):
    print("\n" + "=" * 65)
    print("ANALYSIS 3: Tangible 조건 기관 추출 및 실존 여부 검증")
    print("=" * 65)
    print("목적: 인용된 기관이 실제 존재하는 곳인지 확인\n")

    # Regex-based extraction
    all_orgs = set()
    for turn in all_turns:
        if turn["condition"] != "tangible":
            continue
        all_orgs.update(extract_org_names(turn["content"]))

    print(f"  Regex 추출 후보: {len(all_orgs)}개")

    # LLM-based extraction from a sample (for completeness)
    sample_texts = [t["content"] for t in all_turns if t["condition"] == "tangible"][:15]
    combined = "\n\n---\n\n".join(sample_texts)

    llm_extract = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "다음 텍스트에서 언급된 모든 고유한 조직/기관/단체/기업 이름을 추출하세요.\n"
                "한국어 이름과 영어 이름을 모두 포함하되, 중복 없이 JSON으로 반환하세요.\n"
                '형식: {"orgs": ["이름1", "이름2", ...]}\n\n'
                + combined[:5000]
            )
        }],
        response_format={"type": "json_object"},
    )
    llm_orgs = set(json.loads(llm_extract.choices[0].message.content).get("orgs", []))
    all_orgs = all_orgs | llm_orgs
    all_orgs = {o.strip() for o in all_orgs if len(o.strip()) > 3}

    print(f"  LLM 보완 후 총 후보: {len(all_orgs)}개\n")

    # Verify with GPT-4o
    print(f"  GPT-4o로 실존 여부 검증 중...")
    org_list_str = "\n".join(f"- {o}" for o in sorted(all_orgs))

    verification = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                "아래 목록은 LLM이 동물실험 관련 토론에서 생성한 텍스트에서 추출한 기관/조직명입니다.\n"
                "각 기관에 대해 실제로 존재하는지 판단하고 JSON으로 응답해주세요.\n\n"
                "status 값:\n"
                "  - \"real\": 실제 존재하는 기관 (이름이 약간 다를 수 있어도 실제 기관을 가리키면 real)\n"
                "  - \"fake\": 실제로 존재하지 않는 가상/허구 기관\n"
                "  - \"unclear\": 확인 불가\n\n"
                '응답 형식: {"results": [{"name": "...", "status": "real|fake|unclear", '
                '"official_name": "공식 영문명(알 경우)", "note": "한 줄 설명"}]}\n\n'
                f"기관 목록:\n{org_list_str}"
            )
        }],
        response_format={"type": "json_object"},
    )
    data = json.loads(verification.choices[0].message.content)
    results = data.get("results", [])

    real = [r for r in results if r.get("status") == "real"]
    fake = [r for r in results if r.get("status") == "fake"]
    unclear = [r for r in results if r.get("status") == "unclear"]

    # Print results
    print(f"\n  ✅ 실존 기관 ({len(real)}개):")
    for r in sorted(real, key=lambda x: x["name"]):
        official = r.get("official_name", "")
        note = r.get("note", "")
        suffix = f" → {official}" if official and official.strip() != r["name"] else ""
        print(f"    • {r['name']}{suffix}")
        if note:
            print(f"      └ {note}")

    print(f"\n  ❌ 허구/오류 기관명 ({len(fake)}개):")
    for r in sorted(fake, key=lambda x: x["name"]):
        note = r.get("note", "")
        print(f"    • {r['name']}")
        if note:
            print(f"      └ {note}")

    if unclear:
        print(f"\n  ❓ 불분명 ({len(unclear)}개):")
        for r in sorted(unclear, key=lambda x: x["name"]):
            note = r.get("note", "")
            print(f"    • {r['name']}")
            if note:
                print(f"      └ {note}")

    total = len(results)
    print(f"\n  ▶ 요약: 실존 {len(real)}/{total} ({100*len(real)/total:.0f}%),  "
          f"허구 {len(fake)}/{total} ({100*len(fake)/total:.0f}%),  "
          f"불분명 {len(unclear)}/{total} ({100*len(unclear)/total:.0f}%)")

    return data


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("시뮬레이션 데이터 로딩...")
    authorless = load_jsonl(DATA_DIR / "authorless.jsonl")
    tangible = load_jsonl(DATA_DIR / "tangible.jsonl")
    all_turns = get_assistant_turns(authorless + tangible)
    print(f"  authorless {len(authorless)}개 + tangible {len(tangible)}개 = 총 {len(all_turns)} assistant turns\n")

    sim_results = {}
    sim_results["claim_topics"] = analyze_claim_topics(all_turns, k=6)
    sim_results["org_mention_stats"] = analyze_org_mentions(all_turns)
    sim_results["org_verification"] = analyze_org_verification(all_turns)

    # Save to JSON
    output_path = DATA_DIR / "operationalization_check.json"

    def serialize(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(i) for i in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialize(sim_results), f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_path}")
    print("\n분석 완료.")


if __name__ == "__main__":
    main()
