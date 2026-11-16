"""Build full 172-page multimodal index (text + Vision summaries + pgvector) for Vinamilk AR 2022."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.advanced_scan import normalize_ws, render_pages_winrt  # noqa: E402
from src.retrieval_benchmark import DEFAULT_DATABASE_URL, DEFAULT_EMBEDDING_MODEL  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISUAL_KEYWORDS = (
    "biểu đồ",
    "sơ đồ",
    "cơ cấu",
    "tỷ trọng",
    "tăng trưởng",
    "kết quả nổi bật",
    "tóm tắt thông tin tài chính",
    "mô hình",
    "chuỗi giá trị",
)


def connect_pg(database_url: str) -> Any:
    try:
        import psycopg

        return psycopg.connect(database_url, connect_timeout=15)
    except ImportError:
        import pg8000.dbapi

        parsed = urlparse(database_url)
        return pg8000.dbapi.connect(
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 5432,
            database=(parsed.path or "/").lstrip("/"),
            timeout=15,
        )


def chunk_text(text: str, max_chars: int = 650, overlap: int = 100) -> list[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = text.rfind(" ", start + max_chars // 2, end)
            if split_at > start:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def encode_image_b64(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode("ascii")


def caption_page_vision(
    api_key: str, model: str, img_path: Path, page_num: int, max_retries: int = 3
) -> tuple[str, float]:
    b64 = encode_image_b64(img_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/VinUni-AI20k",
        "X-Title": "Day7-Advanced-Index",
    }
    prompt = (
        f"Đây là trang {page_num} của Báo cáo thường niên Vinamilk 2022. "
        "Hãy mô tả ngắn gọn nhưng chính xác các thành phần trực quan trên trang (nếu có): "
        "1) Biểu đồ (tiêu đề, trục, đơn vị, các giá trị/xu hướng chính); "
        "2) Bảng số liệu (tên bảng, các cột/hàng chính, đơn vị); "
        "3) Ảnh/Sơ đồ (nội dung chính, nhãn/chú thích). "
        "Chỉ ghi thông tin thực sự nhìn thấy trên ảnh, bằng tiếng Việt ngắn gọn (tối đa 180 từ)."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 350,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:250]}")
            data = resp.json()
            usage = data.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            if cost == 0.0:
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                cost = pt * 0.15 / 1_000_000 + ct * 0.60 / 1_000_000
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            return normalize_ws(text), cost
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    return "", 0.0


def upsert_pgvector(
    database_url: str,
    strategy: str,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
) -> None:
    conn = connect_pg(database_url)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_chunks WHERE strategy = %s", (strategy,))
        sql = """
            INSERT INTO document_chunks
            (chunk_id, strategy, document_id, source, page, section, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
        """
        for c, vec in zip(chunks, vectors):
            vec_str = "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"
            cur.execute(
                sql,
                (
                    c["id"],
                    strategy,
                    c["source_id"],
                    c["source"],
                    int(c["page"]),
                    str(c["metadata"].get("chunk_type", "text")),
                    c["text"],
                    json.dumps(c["metadata"], ensure_ascii=False),
                    vec_str,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build 172-page multimodal index for Vinamilk AR 2022.")
    parser.add_argument("--pdf", type=Path, default=Path("data/vinamilk_ar_2022.pdf"))
    parser.add_argument("--pages-dir", type=Path, default=Path("data/advanced_pages"))
    parser.add_argument("--captions-file", type=Path, default=Path("data/advanced_visual_captions.json"))
    parser.add_argument("--chunks-out", type=Path, default=Path("data/advanced_chunks.json"))
    parser.add_argument("--stats-out", type=Path, default=Path("data/advanced_index_stats.json"))
    parser.add_argument("--strategy", default="ADV_VNM2022")
    parser.add_argument("--skip-vision", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    reader = PdfReader(str(args.pdf))
    total_pages = len(reader.pages)

    # 1. Render all pages to PNG in batches of 20 (skipping already rendered)
    missing_pages = [
        p for p in range(1, total_pages + 1) if not (args.pages_dir / f"page_{p:03d}.png").exists()
    ]
    if missing_pages:
        print(f"Rendering {len(missing_pages)} PDF pages to {args.pages_dir}...")
        for i in range(0, len(missing_pages), 20):
            batch = missing_pages[i : i + 20]
            render_pages_winrt(args.pdf, args.pages_dir, batch, dpi_scale=1.5)
            print(f"  Rendered batch {batch[0]}..{batch[-1]}")
    else:
        print(f"All {total_pages} page PNGs already exist in {args.pages_dir}.")

    # 2. Extract page text and identify visual pages
    page_records: list[dict[str, Any]] = []
    visual_pages: list[int] = []
    for p in range(1, total_pages + 1):
        page_obj = reader.pages[p - 1]
        text = normalize_ws(page_obj.extract_text() or "")
        try:
            n_imgs = len(page_obj.images)
        except Exception:
            n_imgs = 0
        has_kw = any(kw in text.lower() for kw in VISUAL_KEYWORDS)
        if n_imgs > 0 or has_kw:
            visual_pages.append(p)
        page_records.append({"page": p, "text": text, "image_count": n_imgs})

    # 3. Vision captioning with checkpointing
    captions: dict[str, str] = {}
    if args.captions_file.exists():
        captions = json.loads(args.captions_file.read_text(encoding="utf-8"))

    vision_cost = 0.0
    vision_errors: list[dict[str, Any]] = []
    if not args.skip_vision:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY for Vision captioning.")
        todo = [p for p in visual_pages if str(p) not in captions]
        print(f"Vision captioning: {len(todo)} pending / {len(visual_pages)} visual pages...")
        for idx, p in enumerate(todo, start=1):
            img_path = args.pages_dir / f"page_{p:03d}.png"
            try:
                cap, cost = caption_page_vision(api_key, model, img_path, p)
                captions[str(p)] = cap
                vision_cost += cost
                if idx % 10 == 0 or idx == len(todo):
                    args.captions_file.write_text(
                        json.dumps(captions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    print(f"  [{idx}/{len(todo)}] p{p:03d} captioned (cum_cost=${vision_cost:.4f})")
            except Exception as exc:
                vision_errors.append({"page": p, "error": str(exc)})
                print(f"  [WARN] p{p:03d} vision error: {exc}")

    # 4. Build chunks (text + visual summary)
    chunks: list[dict[str, Any]] = []
    for rec in page_records:
        p = rec["page"]
        img_rel = f"data/advanced_pages/page_{p:03d}.png"
        for c_idx, piece in enumerate(chunk_text(rec["text"]), start=1):
            chunks.append(
                {
                    "id": f"{args.strategy}_p{p:03d}_c{c_idx:02d}",
                    "text": piece,
                    "source_id": "vinamilk-ar-2022",
                    "source": "data/vinamilk_ar_2022.pdf",
                    "page": p,
                    "metadata": {
                        "chunk_type": "text",
                        "image_path": img_rel,
                        "image_count": rec["image_count"],
                        "strategy": args.strategy,
                    },
                }
            )
        vis_cap = captions.get(str(p), "").strip()
        if vis_cap:
            chunks.append(
                {
                    "id": f"{args.strategy}_p{p:03d}_vis",
                    "text": f"[Mô tả trực quan trang {p}] {vis_cap}",
                    "source_id": "vinamilk-ar-2022",
                    "source": "data/vinamilk_ar_2022.pdf",
                    "page": p,
                    "metadata": {
                        "chunk_type": "visual_summary",
                        "image_path": img_rel,
                        "image_count": rec["image_count"],
                        "strategy": args.strategy,
                    },
                }
            )

    args.chunks_out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 5. Embed and upsert to pgvector
    from sentence_transformers import SentenceTransformer

    emb_model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    print(f"Embedding {len(chunks)} chunks with {emb_model_name}...")
    model_st = SentenceTransformer(emb_model_name)
    vectors = model_st.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vec_list = [v.astype(float).tolist() for v in vectors]
    upsert_pgvector(db_url, args.strategy, chunks, vec_list)

    elapsed_sec = round(time.perf_counter() - t0, 2)
    index_stats = {
        "document": "Vinamilk Annual Report 2022 (vinamilk_ar_2022.pdf)",
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "text_chunks": sum(1 for c in chunks if c["metadata"]["chunk_type"] == "text"),
        "visual_chunks": sum(1 for c in chunks if c["metadata"]["chunk_type"] == "visual_summary"),
        "visual_pages_targeted": len(visual_pages),
        "vision_errors": vision_errors,
        "vision_cost_usd": round(vision_cost, 5),
        "elapsed_sec": elapsed_sec,
        "strategy": args.strategy,
        "embedding_model": emb_model_name,
    }
    args.stats_out.write_text(json.dumps(index_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Index complete:", json.dumps(index_stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
