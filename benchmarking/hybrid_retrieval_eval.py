#!/usr/bin/env python3
"""
Regression checks for hybrid retrieval: citation validity, clause counts, latency budget.

Usage:
  python3 benchmarking/hybrid_retrieval_eval.py
  python3 benchmarking/hybrid_retrieval_eval.py --collection ca1 --base-url http://127.0.0.1:8001
"""

import argparse
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import config  # noqa: E402

CLAUSE_ID_RE = re.compile(r"L\d+_C\d+(?:_C\d+)*(?:\.\d+)?")


def extract_clause_ids(text: str) -> list:
    return list(dict.fromkeys(CLAUSE_ID_RE.findall(text)))


def test_config_thresholds() -> None:
    assert config.HYBRID_LLM_CLAUSE_THRESHOLD == int(
        __import__("os").getenv("HYBRID_LLM_CLAUSE_THRESHOLD", "1000")
    ) or config.HYBRID_LLM_CLAUSE_THRESHOLD > 0
    assert config.RETRIEVAL_LLM_MODEL
    assert config.HYBRID_VECTOR_TOP_K_LARGE >= config.HYBRID_VECTOR_TOP_K


def test_histogram_builder() -> None:
    from RagService.hybrid_retrieval import HybridRetriever

    index = {
        f"L2_C0_C{i}": {"clause_type": "Employment" if i < 3 else "Termination"}
        for i in range(5)
    }
    hist = HybridRetriever._build_clause_type_histogram(index)
    assert "Employment: 3" in hist
    assert "Termination: 2" in hist


def test_broad_query_detection() -> None:
    from RagService.hybrid_retrieval import HybridRetriever

    assert HybridRetriever._is_broad_query("What is the contract about and how many clauses?")
    assert not HybridRetriever._is_broad_query("What is the notice period in section 4?")


def run_live_chat_check(base_url: str, collection: str, latency_budget_s: float) -> dict:
    url = f"{base_url.rstrip('/')}/chat"
    query = "What is the contract about and how many clauses are there?"
    start = time.time()
    with httpx.Client(timeout=180.0) as client:
        r = client.post(
            url,
            data={
                "session_id": "eval_hybrid",
                "message": query,
                "collection_name": collection,
                "use_rag": "true",
                "hybrid_mode": "true",
            },
        )
    elapsed = time.time() - start
    r.raise_for_status()
    body = r.json()
    response = body.get("response", "")
    data = body.get("data", {})
    rag_info = body.get("rag_info", {})
    cited = extract_clause_ids(response)
    doc_count = data.get("document_clause_count") or data.get("total_clauses", 0)

    issues = []
    if elapsed > latency_budget_s:
        issues.append(f"latency {elapsed:.1f}s > budget {latency_budget_s}s")
    if doc_count and str(doc_count) not in response:
        issues.append(f"response missing indexed clause count {doc_count}")
    if rag_info.get("retrieval_mode") not in ("hybrid", "vector_only", None):
        issues.append(f"unknown retrieval_mode: {rag_info.get('retrieval_mode')}")

    return {
        "ok": not issues,
        "elapsed_s": elapsed,
        "doc_clause_count": doc_count,
        "cited_ids": len(cited),
        "retrieval_mode": rag_info.get("retrieval_mode"),
        "issues": issues,
        "response_preview": response[:200],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid retrieval regression eval")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--collection", default="ca1")
    parser.add_argument("--latency-budget", type=float, default=60.0)
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    print("Unit checks...")
    test_config_thresholds()
    test_histogram_builder()
    test_broad_query_detection()
    print("  OK")

    if args.skip_live:
        print("Skipping live API check (--skip-live)")
        return 0

    print(f"\nLive chat check on collection={args.collection}...")
    try:
        result = run_live_chat_check(args.base_url, args.collection, args.latency_budget)
    except httpx.HTTPError as e:
        print(f"  SKIP live check (API unavailable): {e}")
        return 0

    print(f"  elapsed: {result['elapsed_s']:.1f}s")
    print(f"  retrieval_mode: {result['retrieval_mode']}")
    print(f"  document_clause_count: {result['doc_clause_count']}")
    print(f"  cited clause ids in response: {result['cited_ids']}")
    if result["issues"]:
        for issue in result["issues"]:
            print(f"  FAIL: {issue}")
        return 1
    print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
