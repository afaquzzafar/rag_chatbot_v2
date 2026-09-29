# ==============================================================================
# frontend/app.py
# ------------------------------------------------------------------------------
# STEP: "Display in Streamlit" -- the ONLY user-facing surface of this app.
#
# Run with:  streamlit run frontend/app.py
#
# WHAT THIS FILE DOES
#   A chat-only Streamlit UI: no document upload widget anywhere (per the
#   requirement that all ingestion happens in the backend). On first load,
#   it silently runs backend ingestion if the vector store is empty, then
#   presents a plain chat interface.
#
# WHY st.cache_resource FOR THE PIPELINE
#   Streamlit re-runs this entire script top-to-bottom on every user
#   interaction (typing a message, clicking a button). Without caching,
#   we'd reload the Gemini client and re-run ingestion on every single
#   keystroke-triggered rerun. `st.cache_resource` tells Streamlit "build
#   this once per session/process and hand back the same object every
#   rerun" -- the correct caching tool for stateful clients/connections
#   (as opposed to `st.cache_data`, which is for cacheable *data*).
#
# WHY st.session_state FOR CHAT HISTORY
#   Streamlit has no built-in notion of "conversation" -- every rerun is a
#   fresh script execution. `st.session_state` is Streamlit's per-browser-
#   session dictionary that persists across reruns, which is where we keep
#   the displayed message list (separate from RAGPipeline's own internal
#   ConversationMemory, which is used for LLM context/query rewriting).
#
# STREAMING
#   A new answer is displayed token-by-token instead of appearing all at
#   once: pipeline.answer_question() takes an on_token callback (see
#   rag_pipeline/rag_pipeline.py) that this file uses to re-render a
#   st.empty() placeholder with the accumulated text on every new piece.
#   Only the LATEST answer streams -- past messages replayed from
#   st.session_state on a rerun render instantly, exactly as before.
#
# INPUT / OUTPUT
#   Input:  user's typed question via st.chat_input.
#   Output: rendered chat bubbles, expandable source citations, a
#           confidence score, and a sidebar with model info.
# ==============================================================================

import sys
from pathlib import Path

# Allow `streamlit run frontend/app.py` to import sibling packages
# (config, rag_pipeline, scripts, ...) as if run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from config.settings import settings
from rag_pipeline.rag_pipeline import RAGPipeline
from scripts.ingest import run_ingestion
from utils.feedback_logger import log_feedback
from vector_store import chroma_manager

st.set_page_config(
    page_title="Healthcare Insurance Assistant",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    """Build the RAGPipeline once per process (Gemini client, embeddings, etc.)."""
    settings.validate()
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def ensure_documents_ingested() -> dict:
    """
    Auto-ingest the backend's PDF documents on first load if the vector
    store is empty (or new/changed PDFs are detected) -- this is the only
    place ingestion happens; there is deliberately no upload widget in this
    UI, per the "documents live and are managed entirely in the backend"
    requirement.

    Cached with st.cache_resource so this only actually runs once per
    process/session, not on every chat rerun.
    """
    return run_ingestion()


def render_sidebar() -> None:
    """Render the sidebar (model info, retrieved doc count)."""
    with st.sidebar:
        st.header("Healthcare Insurance Assistant")
        st.caption("Ask about your coverage, benefits, and plan documents.")

        st.subheader("Model Info")
        chat_model_label = (
            settings.gemini_chat_model
            if settings.llm_provider == "gemini"
            else settings.databricks_llm_endpoint
        )
        st.text(f"LLM provider: {settings.llm_provider}")
        st.text(f"Chat model: {chat_model_label}")
        st.text(f"Embedding provider: {settings.embedding_provider}")
        st.text(f"Top-K retrieved: {settings.top_k}")
        st.text(f"Hybrid search: {'on' if settings.enable_hybrid_search else 'off'}")

        st.subheader("Knowledge Base")
        try:
            chunk_count = chroma_manager.count()
        except Exception:
            chunk_count = "unavailable"
        st.text(f"Indexed chunks: {chunk_count}")

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            load_pipeline().memory.clear()
            st.rerun()


def render_sources(sources: list) -> None:
    """Render an expandable citation panel for one assistant message."""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**{i}. {source['document_name']}** — page {source['page_number']} "
                f"_({source['document_type']})_"
            )
            if source.get("section_title"):
                # Only chapter/section-structured documents (e.g. the
                # Evidence of Coverage) set this -- see chunking/structure_chunker.py.
                st.caption(f"{source.get('chapter_title', '')} › {source['section_title']}")
            st.progress(min(max(source["score"], 0.0), 1.0), text=f"Relevance score: {source['score']:.2f}")


def render_assistant_message(index: int, message: dict, skip_content: bool = False) -> None:
    """
    Render one assistant turn: answer text, sources, confidence, and the
    👍/👎 feedback buttons -- shared by both the chat-history replay loop and
    the just-generated answer, so feedback buttons appear immediately after
    a response is generated instead of only on the NEXT rerun (the history
    loop is the only place that used to render them, and Streamlit doesn't
    re-run that loop until some later interaction triggers a rerun).

    `index` must be the message's stable position in st.session_state.messages
    -- used as the button keys' suffix so they stay unique and consistent
    across reruns (the identical index the history loop will use for this
    same message on the next rerun).

    `skip_content=True` for a just-streamed answer: its text was already
    rendered incrementally into a placeholder as it arrived (see main()), so
    rendering it again here would show it twice.
    """
    if message.get("service_error"):
        # The AI service failed (quota, invalid key, outage): no answer was
        # generated, so there are no sources to cite or answer to rate --
        # just the error, with 0% confidence.
        if not skip_content:
            st.warning(message["content"], icon="⚠️")
        st.caption("Confidence: 0%")
        return

    if not skip_content:
        st.markdown(message["content"])
    render_sources(message.get("sources", []))
    if message.get("confidence") is not None:
        st.caption(f"Confidence: {message['confidence']:.0%}")
    if message.get("grounded") is False:
        st.warning("Grounding check flagged this answer as possibly not fully supported by the retrieved context.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👍", key=f"up_{index}"):
            log_feedback(
                message.get("question", ""),
                message["content"],
                message.get("sources", []),
                rating="up",
            )
            st.toast("Thanks for the feedback!", icon="👍")
    with col2:
        if st.button("👎", key=f"down_{index}"):
            log_feedback(
                message.get("question", ""),
                message["content"],
                message.get("sources", []),
                rating="down",
            )
            st.toast("Thanks — we'll use this to improve.", icon="👎")


def main() -> None:
    render_sidebar()

    st.title("🩺 Healthcare Insurance Assistant")
    st.caption(
        "I answer questions using your official Evidence of Coverage, Summary of "
        "Benefits, and plan documents. I only answer from those documents and "
        "always cite my sources."
    )

    with st.spinner("Indexing insurance documents (first run only)..."):
        ingestion_summary = ensure_documents_ingested()
    if ingestion_summary.get("files_processed"):
        st.toast(
            f"Indexed {len(ingestion_summary['files_processed'])} document(s), "
            f"{ingestion_summary['total_chunks']} chunks.",
            icon="✅",
        )

    pipeline = load_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(i, message)
            else:
                st.markdown(message["content"])

    question = st.chat_input("Ask about your coverage, deductibles, benefits...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # A placeholder that first shows a "searching" notice, then gets
            # overwritten with the answer as it streams in -- generate_answer()
            # only calls on_token() once retrieval/multi-hop are
            # already done and the LLM call has actually started, so this
            # notice naturally covers that earlier work with no separate
            # spinner needed.
            placeholder = st.empty()
            placeholder.markdown("_Searching your plan documents..._")

            def on_token(accumulated_text: str) -> None:
                placeholder.markdown(accumulated_text + "▌")

            result = pipeline.answer_question(question, on_token=on_token)
            if result["service_error"]:
                placeholder.warning(result["answer"], icon="⚠️")
            else:
                placeholder.markdown(result["answer"])

            # Append BEFORE rendering, so the message's index in
            # st.session_state.messages is already final -- the feedback
            # buttons' keys then match what the history loop above will use
            # for this exact message on every future rerun.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "confidence": result["confidence"],
                    "grounded": result["grounded"],
                    "service_error": result["service_error"],
                    "question": question,
                }
            )
            new_index = len(st.session_state.messages) - 1
            render_assistant_message(new_index, st.session_state.messages[new_index], skip_content=True)


if __name__ == "__main__":
    main()
