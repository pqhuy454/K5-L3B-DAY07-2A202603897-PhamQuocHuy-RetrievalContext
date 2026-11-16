"""ChatGPT/Gemini-style Streamlit Demo UI for Day 7 Agentic RAG (Basic + Advanced Vision)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.advanced_agent import run_case as run_advanced_case  # noqa: E402
from src.agent_runner import run_single_case as run_basic_case  # noqa: E402
from src.retrieval_benchmark import (  # noqa: E402
    DEFAULT_DATABASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DensePgVectorRetriever,
    FilteredTfidfRetriever,
    HybridRrfRetriever,
)

load_dotenv()

st.set_page_config(
    page_title="Day 7 — Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang nạp Embedding model & kết nối pgvector Day 6...")
def load_resources(mode: str) -> dict[str, Any]:
    db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    emb_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    if mode == "basic":
        chunks = json.loads((ROOT_DIR / "data/day6_chunks.json").read_text(encoding="utf-8"))
        lexical = FilteredTfidfRetriever(chunks)
        dense = DensePgVectorRetriever(database_url=db_url, model_name=emb_model, strategy="B")
        hybrid = HybridRrfRetriever(dense=dense, lexical=lexical, candidate_k=10, rrf_k=60)
        return {"chunks": chunks, "lexical": lexical, "hybrid": hybrid}

    chunks = json.loads((ROOT_DIR / "data/advanced_chunks.json").read_text(encoding="utf-8"))
    chunks_by_page: dict[int, list[dict[str, Any]]] = {}
    for c in chunks:
        chunks_by_page.setdefault(int(c["page"]), []).append(c)
    lexical = FilteredTfidfRetriever(chunks)
    dense = DensePgVectorRetriever(database_url=db_url, model_name=emb_model, strategy="ADV_VNM2022")
    hybrid = HybridRrfRetriever(dense=dense, lexical=lexical, candidate_k=12, rrf_k=60)
    return {"chunks": chunks, "chunks_by_page": chunks_by_page, "hybrid": hybrid}


def render_assistant_details(msg: dict[str, Any], mode: str) -> None:
    status = msg.get("status", "")
    if status == "abstain":
        st.warning("Abstain: Tài liệu không đủ bằng chứng để trả lời câu hỏi này.")
    elif status == "error":
        st.error(f"Lỗi hệ thống: {msg.get('error')}")

    events = msg.get("events") or []
    citations = msg.get("citations") or []
    viewed = msg.get("viewed_pages") or []
    meta_str = f"⏱️ {msg.get('elapsed_ms', 0):.0f} ms · 💰 ${msg.get('cost_usd', 0):.5f}"

    with st.expander(f"🔎 Quá trình truy xuất & Nguồn trích dẫn ({len(events)} bước · {meta_str})"):
        if events:
            st.markdown("**Tool calls:**")
            for idx, ev in enumerate(events, start=1):
                st.json(ev, expanded=False)
        if citations:
            st.markdown("**Citations:**")
            st.write(citations)
        if mode == "advanced" and viewed:
            st.markdown("**Trang PDF gốc Agent đã quan sát trực tiếp:**")
            cols = st.columns(min(len(viewed), 2))
            for idx, page_num in enumerate(viewed):
                img_path = ROOT_DIR / "data/advanced_pages" / f"page_{int(page_num):03d}.png"
                if img_path.exists():
                    cols[idx % len(cols)].image(
                        str(img_path),
                        caption=f"Trang {page_num} (vinamilk_ar_2022.pdf)",
                        use_container_width=True,
                    )


def render_benchmark_view() -> None:
    st.subheader("📊 Tổng hợp Benchmark Day 7")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Phần cơ bản — Retrieval (`dev` & `test`)")
        for p in ("results/retrieval_dev/summary.json", "results/retrieval_test/summary.json"):
            fp = ROOT_DIR / p
            if fp.exists():
                st.caption(p)
                st.dataframe(json.loads(fp.read_text(encoding="utf-8")), use_container_width=True)

        st.markdown("#### Phần cơ bản — Agent (`test`)")
        for p in ("results/agent_baseline_test/summary.json", "results/agent_hybrid_multi_test/summary.json"):
            fp = ROOT_DIR / p
            if fp.exists():
                st.caption(p)
                st.json(json.loads(fp.read_text(encoding="utf-8")))

    with col2:
        st.markdown("#### Thử thách nâng cao — Vinamilk AR 2022 (`test`, 30 câu)")
        rev_path = ROOT_DIR / "results/advanced_test/reviews.json"
        if rev_path.exists():
            rows = json.loads(rev_path.read_text(encoding="utf-8"))
            passed = sum(1 for r in rows if r.get("pass"))
            st.metric("Tỷ lệ đạt toàn bộ test", f"{passed}/{len(rows)} ({passed / len(rows) * 100:.1f}%)")
            st.dataframe(rows, use_container_width=True)


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    with st.sidebar:
        st.title("🤖 Agentic RAG")
        mode_label = st.radio(
            "Chọn hệ thống demo",
            [
                "Luật An ninh mạng 2018 (Cơ bản)",
                "Vinamilk AR 2022 Vision (Nâng cao)",
            ],
        )
        mode = "basic" if "Cơ bản" in mode_label else "advanced"
        view_mode = st.segmented_control("Chế độ xem", ["💬 Chat", "📊 Benchmark"], default="💬 Chat")

        if mode == "basic":
            st.caption("Corpus: 165 chunks (32 trang) · Model: `openai/gpt-4o-mini`")
            samples = [
                "Ai có thẩm quyền ban hành Danh mục hệ thống thông tin quan trọng về an ninh quốc gia?",
                "Luật An ninh mạng có hiệu lực từ ngày nào và thời hạn bảo đảm điều kiện là bao lâu?",
                "Mức phạt tiền cụ thể bằng đồng đối với vi phạm lưu trữ dữ liệu là bao nhiêu?",
            ]
        else:
            st.caption("Corpus: 819 chunks (172 trang + PNG) · Model: `google/gemini-2.5-flash`")
            samples = [
                "Theo biểu đồ Doanh thu (Tỷ Đồng), doanh thu năm 2021 và 2022 là bao nhiêu?",
                "Theo sơ đồ Mô hình hoạt động, tỷ lệ sở hữu 9% và 59% là tại công ty nào?",
                "Ngân sách R&D dự kiến riêng cho năm 2024 của Vinamilk là bao nhiêu tỷ đồng?",
            ]

        st.markdown("**Câu hỏi gợi ý nhanh:**")
        for idx, s in enumerate(samples):
            if st.button(s, key=f"sample_{mode}_{idx}", use_container_width=True):
                st.session_state.pending_prompt = s

        if st.button("🗑️ Cuộc trò chuyện mới", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if view_mode == "📊 Benchmark":
        render_benchmark_view()
        return

    st.markdown(f"### 💬 {mode_label}")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_assistant_details(m, m.get("mode", mode))

    user_input = st.chat_input("Nhập câu hỏi của bạn...")
    prompt = st.session_state.pending_prompt or user_input
    st.session_state.pending_prompt = None

    if not prompt:
        return

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        st.error("Thiếu `OPENROUTER_API_KEY` trong file `.env`.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    resources = load_resources(mode)
    with st.chat_message("assistant"):
        with st.status("Agent đang tìm kiếm và đối chiếu bằng chứng...", expanded=True) as status_box:
            if mode == "basic":
                model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
                res = run_basic_case(
                    case={"id": "demo", "split": "demo", "question": prompt},
                    api_key=api_key,
                    model=model,
                    lexical=resources["lexical"],
                    hybrid=resources["hybrid"],
                    allow_hybrid=True,
                    max_searches=3,
                    budget_chars=2400,
                    top_k=5,
                )
            else:
                model = os.getenv("ADVANCED_MODEL", "google/gemini-2.5-flash").strip()
                res = run_advanced_case(
                    case={"id": "demo", "split": "demo", "category": "Demo", "question": prompt},
                    api_key=api_key,
                    model=model,
                    hybrid=resources["hybrid"],
                    chunks_by_page=resources["chunks_by_page"],
                    pages_dir=ROOT_DIR / "data/advanced_pages",
                )
            status_box.update(label="Hoàn tất truy xuất!", state="complete", expanded=False)

        answer_text = res.get("answer") or "(Không có nội dung trả lời)"
        st.markdown(answer_text)
        msg_obj = {
            "role": "assistant",
            "mode": mode,
            "content": answer_text,
            "status": res.get("status"),
            "citations": res.get("citations", []),
            "events": res.get("events", []),
            "viewed_pages": res.get("viewed_pages", []),
            "elapsed_ms": res.get("elapsed_ms", 0),
            "cost_usd": res.get("cost_usd", 0),
            "error": res.get("error"),
        }
        render_assistant_details(msg_obj, mode)
        st.session_state.messages.append(msg_obj)


if __name__ == "__main__":
    main()
