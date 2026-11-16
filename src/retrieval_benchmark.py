"""Benchmark 3+ retrieval methods and improvement configs on Day 6 Strategy B corpus."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from starter.baseline import TfidfIndex, hit, pack_context  # noqa: E402

DEFAULT_DATABASE_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PERMITTED_FILTER_KEYS = {
    "section",
    "year",
    "document_type",
    "effective_date",
    "language",
    "document_id",
    "page",
}


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def connect_pg(database_url: str) -> Any:
    try:
        import psycopg

        return psycopg.connect(database_url, connect_timeout=10)
    except ImportError:
        import pg8000.dbapi

        parsed = urlparse(database_url)
        return pg8000.dbapi.connect(
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 5432,
            database=(parsed.path or "/").lstrip("/"),
            timeout=10,
        )


def matches_filter(chunk: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    if not flt:
        return True
    meta = chunk.get("metadata") or {}
    for k, v in flt.items():
        if k not in PERMITTED_FILTER_KEYS:
            continue
        if k == "document_id":
            if chunk.get("source_id") != v:
                return False
        elif k == "page":
            if int(chunk.get("page", -1)) != int(v):
                return False
        elif str(meta.get(k, "")) != str(v):
            return False
    return True


class FilteredTfidfRetriever:
    """Lexical TF-IDF retriever wrapping starter.baseline.TfidfIndex."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(
        self, query: str, top_k: int = 3, flt: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not flt:
            return self.index.search(query, top_k=top_k)
        # Score all matches then filter by permitted metadata keys
        candidates = self.index.search(query, top_k=len(self.index.chunks))
        return [c for c in candidates if matches_filter(c, flt)][:top_k]


class DensePgVectorRetriever:
    """Dense retriever querying Day 6 PostgreSQL + pgvector (strategy B)."""

    def __init__(self, database_url: str, model_name: str, strategy: str = "B") -> None:
        from sentence_transformers import SentenceTransformer

        self.database_url = database_url
        self.model_name = model_name
        self.strategy = strategy
        self._model = SentenceTransformer(model_name)

    def search(
        self, query: str, top_k: int = 3, flt: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        vec = self._model.encode(
            query.strip(),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vec_str = "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"

        where_clauses = ["strategy = %s"]
        params: list[Any] = [vec_str, self.strategy]

        if flt:
            for k, v in flt.items():
                if k not in PERMITTED_FILTER_KEYS:
                    continue
                if k in ("section", "document_id"):
                    where_clauses.append(f"{k} = %s")
                    params.append(str(v))
                elif k == "page":
                    where_clauses.append("page = %s")
                    params.append(int(v))
                else:
                    where_clauses.append("metadata->>%s = %s")
                    params.extend([k, str(v)])

        params.extend([vec_str, int(top_k)])
        sql = f"""
            SELECT chunk_id, document_id, source, page, section, text, metadata,
                   1.0 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            WHERE {' AND '.join(where_clauses)}
            ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
            LIMIT %s
        """

        conn = connect_pg(self.database_url)
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        finally:
            conn.close()

        results: list[dict[str, Any]] = []
        for chunk_id, doc_id, source, page, section, text, metadata, score in rows:
            meta = json.loads(metadata) if isinstance(metadata, str) else dict(metadata or {})
            meta.setdefault("section", section)
            results.append(
                {
                    "id": chunk_id,
                    "text": normalize_ws(text),
                    "source_id": doc_id,
                    "source": source,
                    "page": int(page),
                    "metadata": meta,
                    "score": float(score),
                    "strategy": "dense_pgvector",
                }
            )
        return results


class HybridRrfRetriever:
    """Hybrid retriever combining Dense (pgvector) and Lexical (TF-IDF) via RRF."""

    def __init__(
        self,
        dense: DensePgVectorRetriever,
        lexical: FilteredTfidfRetriever,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(
        self, query: str, top_k: int = 3, flt: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        dense_res = self.dense.search(query, top_k=self.candidate_k, flt=flt)
        lex_res = self.lexical.search(query, top_k=self.candidate_k, flt=flt)

        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_res, start=1):
            cid = item["id"]
            by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, item in enumerate(lex_res, start=1):
            cid = item["id"]
            by_id.setdefault(cid, item)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked_ids = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))[:top_k]
        return [
            {**by_id[cid], "score": scores[cid], "strategy": "hybrid_rrf"}
            for cid in ranked_ids
        ]


def context_satisfies_gold(
    context_ids: list[str],
    results_by_id: dict[str, dict[str, Any]],
    gold: list[dict[str, Any]],
    required_count: int,
) -> bool:
    """Check whether packed context chunks satisfy at least required_count gold items."""
    selected = [results_by_id[cid] for cid in context_ids if cid in results_by_id]
    matched_gold = sum(1 for g in gold if any(hit(c, [g]) for c in selected))
    return matched_gold >= required_count


def evaluate_config(
    config_name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    top_k: int,
    budget: int,
    use_filter: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    errors = 0

    for q in questions:
        flt = q.get("filter") if use_filter else None
        t0 = time.perf_counter()
        err_msg: str | None = None
        try:
            results = retriever.search(q["question"], top_k=top_k, flt=flt)
        except Exception as exc:
            results = []
            err_msg = str(exc)
            errors += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Evaluate ranking on top-3 regardless of top_k for fair comparison
        top3 = results[:3]
        context, context_ids = pack_context(results, budget=budget)
        by_id = {c["id"]: c for c in results}
        req_count = int(q.get("required_gold_count", 1))

        marked_top3 = [
            {
                "id": c["id"],
                "source_id": c["source_id"],
                "page": c["page"],
                "rank": idx + 1,
                "score": round(float(c.get("score", 0.0)), 6),
                "hit": hit(c, q["gold"]),
            }
            for idx, c in enumerate(top3)
        ]

        hit1 = bool(marked_top3 and marked_top3[0]["hit"])
        hit3 = any(item["hit"] for item in marked_top3)
        ctx_hit = context_satisfies_gold(context_ids, by_id, q["gold"], req_count)

        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "config": config_name,
                "question": q["question"],
                "filter_used": flt,
                "top_k": top_k,
                "budget_chars": budget,
                "top3": marked_top3,
                "context_ids": context_ids,
                "context_chars": len(context),
                "approx_context_tokens": len(context) // 4,
                "elapsed_ms": round(elapsed_ms, 2),
                "hit1": hit1,
                "hit3": hit3,
                "context_hit": ctx_hit,
                "error": err_msg,
            }
        )

    n = len(questions)
    hit1_n = sum(1 for t in traces if t["hit1"])
    hit3_n = sum(1 for t in traces if t["hit3"])
    ctx_n = sum(1 for t in traces if t["context_hit"])
    latencies = [t["elapsed_ms"] for t in traces]

    summary = {
        "config": config_name,
        "split": questions[0]["split"] if questions else "",
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "use_filter": use_filter,
        "hit1": f"{hit1_n}/{n} ({hit1_n / n * 100:.1f}%)" if n else "0/0",
        "hit3": f"{hit3_n}/{n} ({hit3_n / n * 100:.1f}%)" if n else "0/0",
        "context_evidence_rate": f"{ctx_n}/{n} ({ctx_n / n * 100:.1f}%)" if n else "0/0",
        "hit1_count": hit1_n,
        "hit3_count": hit3_n,
        "context_hit_count": ctx_n,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0.0,
        "median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "errors": errors,
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods on Day 7.")
    parser.add_argument("--chunks", type=Path, default=Path("data/day6_chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/questions_retrieval.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "tfidf_b800",
            "dense_b800",
            "hybrid_rrf_b800",
            "hybrid_rrf_b1600",
            "hybrid_rrf_filter_b1600",
        ],
    )
    args = parser.parse_args()

    output_dir = args.output or Path("results") / f"retrieval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q.get("split") == args.split]

    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    print(f"Loading TF-IDF index ({len(chunks)} chunks) and Dense model ({embedding_model})...")
    lexical = FilteredTfidfRetriever(chunks)
    dense = DensePgVectorRetriever(database_url=database_url, model_name=embedding_model, strategy="B")
    hybrid = HybridRrfRetriever(dense=dense, lexical=lexical, candidate_k=10, rrf_k=60)

    registry: dict[str, tuple[Any, int, int, bool]] = {
        "tfidf_b800": (lexical, 3, 800, False),
        "dense_b800": (dense, 3, 800, False),
        "hybrid_rrf_b800": (hybrid, 3, 800, False),
        "hybrid_rrf_b1600": (hybrid, 3, 1600, False),
        "hybrid_rrf_filter_b1600": (hybrid, 3, 1600, True),
    }

    summaries: list[dict[str, Any]] = []
    for cfg in args.configs:
        if cfg not in registry:
            raise ValueError(f"Unknown config: {cfg}. Available: {list(registry.keys())}")
        retriever, top_k, budget, use_filter = registry[cfg]
        summary, traces = evaluate_config(cfg, retriever, questions, top_k, budget, use_filter)
        summaries.append(summary)
        (output_dir / f"traces_{cfg}.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n| Config | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Avg Chars | Median (ms) | Errors |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        print(
            f"| {s['config']} | {s['split']} | {s['hit1']} | {s['hit3']} | "
            f"{s['context_evidence_rate']} | {s['avg_context_chars']} | {s['median_ms']} | {s['errors']} |"
        )
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
