"""Export Strategy B chunks and benchmark questions from Day 6 to Day 7 JSON schemas."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_DATABASE_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
DEFAULT_DAY6_QUESTIONS = (
    Path(__file__).resolve().parents[2]
    / "K5-L3B-DAY06-PhamQuocHuy-2A202603897-DataIngestion"
    / "benchmark"
    / "questions.json"
)

UNANSWERABLE_CASES: list[dict[str, Any]] = [
    {
        "id": "u01",
        "split": "dev",
        "question": (
            "Theo Luật An ninh mạng 2018, mức phạt tiền cụ thể (tính bằng đồng) đối với "
            "hành vi vi phạm hành chính về lưu trữ dữ liệu tại Việt Nam là bao nhiêu?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Luật An ninh mạng 2018 chỉ quy định nguyên tắc xử lý vi phạm (Điều 9) và giao "
            "Chính phủ quy định chi tiết, không quy định mức phạt tiền cụ thể bằng đồng."
        ),
        "verification_method": (
            "Kiểm tra toàn bộ 165 chunks với từ khóa 'phạt tiền', 'triệu đồng', 'nghị định' "
            "đều trả về 0 kết quả; Điều 9 không có khung tiền phạt."
        ),
    },
    {
        "id": "u02",
        "split": "dev",
        "question": (
            "Luật An ninh mạng 2018 quy định như thế nào về việc cấp phép và quản lý "
            "sàn giao dịch tiền điện tử (cryptocurrency) hoặc tài sản mã hóa?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Luật An ninh mạng 2018 không có điều khoản nào điều chỉnh sàn giao dịch "
            "tiền điện tử, tiền ảo hay tài sản mã hóa."
        ),
        "verification_method": (
            "Kiểm tra toàn bộ 165 chunks với từ khóa 'tiền ảo', 'tiền điện tử', "
            "'tài sản mã hóa', 'bitcoin' đều trả về 0 kết quả."
        ),
    },
    {
        "id": "u03",
        "split": "test",
        "question": (
            "Theo Luật An ninh mạng 2018, tổ chức, cá nhân phải tuân thủ tiêu chuẩn bảo mật "
            "quốc tế ISO/IEC 27001 hoặc quy định GDPR của Liên minh châu Âu trong thời hạn bao lâu?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Luật An ninh mạng 2018 không đề cập đến tiêu chuẩn ISO/IEC 27001 hay quy định GDPR."
        ),
        "verification_method": (
            "Kiểm tra toàn bộ 165 chunks với từ khóa 'iso', '27001', 'gdpr' đều trả về 0 kết quả."
        ),
    },
    {
        "id": "u04",
        "split": "test",
        "question": (
            "Luật An ninh mạng 2018 có quy định mức thuế suất ưu đãi hoặc chính sách miễn thuế "
            "thu nhập doanh nghiệp cho doanh nghiệp cung cấp dịch vụ an ninh mạng không, "
            "nếu có là bao nhiêu phần trăm?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Luật An ninh mạng 2018 không quy định mức thuế suất ưu đãi hay chính sách "
            "miễn thuế thu nhập doanh nghiệp."
        ),
        "verification_method": (
            "Kiểm tra toàn bộ 165 chunks với từ khóa 'thuế' trả về 0 kết quả."
        ),
    },
]


def normalize_ws(text: str) -> str:
    """Collapse PDF line breaks and repeated whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def connect_pg(database_url: str) -> Any:
    """Connect to PostgreSQL using psycopg if available, else pg8000."""
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


def fetch_day6_chunks(database_url: str, strategy: str = "B") -> list[dict[str, Any]]:
    """Load chunks for the selected Day 6 strategy and convert to Day 7 schema."""
    conn = connect_pg(database_url)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_id, document_id, source, page, section, text, metadata "
            "FROM document_chunks WHERE strategy = %s ORDER BY page, chunk_id",
            (strategy,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    chunks: list[dict[str, Any]] = []
    for chunk_id, document_id, source, page, section, text, metadata in rows:
        meta = json.loads(metadata) if isinstance(metadata, str) else dict(metadata or {})
        meta.setdefault("section", section)
        meta.setdefault("strategy", strategy)
        chunks.append(
            {
                "id": chunk_id,
                "text": normalize_ws(text),
                "source_id": document_id,
                "source": source,
                "page": int(page),
                "metadata": meta,
            }
        )
    return chunks


def convert_questions(day6_questions_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert Day 6 questions into retrieval benchmark and full 24-case agent benchmark."""
    raw = json.loads(day6_questions_path.read_text(encoding="utf-8"))
    raw_questions = raw["questions"] if isinstance(raw, dict) else raw

    retrieval_questions: list[dict[str, Any]] = []
    agent_cases: list[dict[str, Any]] = []

    for q in raw_questions:
        evidence_raw = q["evidence_quote"]
        # q20 spans two consecutive clauses/chunks on page 32; split so each gold fits one chunk.
        if "\n" in evidence_raw.strip():
            parts = [normalize_ws(p) for p in evidence_raw.splitlines() if normalize_ws(p)]
            gold = [
                {"source_id": q["gold_source_id"], "page": int(q["gold_page"]), "evidence": part}
                for part in parts
            ]
            required_gold_count = len(gold)
        else:
            gold = [
                {
                    "source_id": q["gold_source_id"],
                    "page": int(q["gold_page"]),
                    "evidence": normalize_ws(evidence_raw),
                }
            ]
            required_gold_count = 1

        ret_item: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "type": q.get("type", "general"),
            "question": q["question"],
            "gold": gold,
            "required_gold_count": required_gold_count,
            "expected_answer": q["expected_answer"],
        }
        if q.get("filter") is not None:
            ret_item["filter"] = q["filter"]
        retrieval_questions.append(ret_item)

        case_item: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": gold,
            "required_gold_count": required_gold_count,
        }
        if q.get("filter") is not None:
            case_item["filter"] = q["filter"]
        agent_cases.append(case_item)

    agent_cases.extend(UNANSWERABLE_CASES)
    return retrieval_questions, agent_cases


def verify_gold_coverage(chunks: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    """Verify every gold evidence snippet is a substring of a chunk on matching source_id/page."""
    for q in questions:
        for g in q["gold"]:
            matched = any(
                c["source_id"] == g["source_id"]
                and c["page"] == g["page"]
                and g["evidence"] in c["text"]
                for c in chunks
            )
            if not matched:
                raise ValueError(
                    f"Gold evidence for {q['id']} (page {g['page']}) not found in any Strategy B chunk."
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Day 6 Strategy B chunks & questions for Day 7.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--strategy", default="B")
    parser.add_argument("--day6-questions", type=Path, default=DEFAULT_DAY6_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    chunks = fetch_day6_chunks(args.database_url, strategy=args.strategy)
    if not chunks:
        raise RuntimeError(f"No chunks found for strategy '{args.strategy}' in PostgreSQL.")

    retrieval_questions, agent_cases = convert_questions(args.day6_questions)
    verify_gold_coverage(chunks, retrieval_questions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "day6_chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "questions_retrieval.json").write_text(
        json.dumps(retrieval_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "agent_cases.json").write_text(
        json.dumps(agent_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dev_ans = sum(1 for c in agent_cases if c["split"] == "dev" and c["answerable"])
    dev_unans = sum(1 for c in agent_cases if c["split"] == "dev" and not c["answerable"])
    test_ans = sum(1 for c in agent_cases if c["split"] == "test" and c["answerable"])
    test_unans = sum(1 for c in agent_cases if c["split"] == "test" and not c["answerable"])
    print(
        f"Exported {len(chunks)} chunks (strategy {args.strategy}) -> {args.output_dir / 'day6_chunks.json'}\n"
        f"Exported {len(retrieval_questions)} retrieval questions -> {args.output_dir / 'questions_retrieval.json'}\n"
        f"Exported {len(agent_cases)} agent cases -> {args.output_dir / 'agent_cases.json'}\n"
        f"Split summary: dev={dev_ans} answerable + {dev_unans} unanswerable | "
        f"test={test_ans} answerable + {test_unans} unanswerable\n"
        f"Gold coverage check: PASS (all gold snippets verified in chunks)"
    )


if __name__ == "__main__":
    main()
