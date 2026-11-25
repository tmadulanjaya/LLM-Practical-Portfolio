import streamlit as st

import rag_core as rc

st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="wide")


@st.cache_resource
def get_vs():
    return rc.get_vectorstore()


def show_sources(cards: list[dict]) -> None:
    with st.expander(f"📎 Sources ({len(cards)} passages)"):
        for c in cards:
            st.markdown(f"**{c['label']}**  ·  distance `{c['distance']}`")
            st.caption(c["snippet"])


# ------------------------------------------------------------------ startup
ok, msg = rc.check_ollama()
if not ok:
    st.error(msg)
    st.stop()

vs = get_vs()
ss = st.session_state
ss.setdefault("messages", [])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📚 Knowledge base")

    files = st.file_uploader("Upload your own PDFs", type="pdf", accept_multiple_files=True)
    if st.button("➕ Add to knowledge base", disabled=not files, type="primary"):
        bar = st.progress(0.0, text="Indexing...")
        for i, f in enumerate(files):
            try:
                n = rc.ingest_pdf(vs, f.getvalue(), f.name)
                st.info(f"{f.name}: already indexed") if n == 0 else st.success(f"{f.name}: {n} chunks")
            except Exception as e:
                st.error(f"{f.name}: {e}")
            bar.progress((i + 1) / len(files), text=f"Indexed {i + 1}/{len(files)}")
        bar.empty()

    if st.button("🔄 Sync ./data folder"):
        with st.spinner("Scanning data folder..."):
            for name, status in rc.ingest_folder(vs).items():
                st.write(f"{name}: {status}")

    st.divider()
    docs = rc.list_documents(vs)
    selected: list[str] = []
    if docs:
        selected = st.multiselect(
            "Search only in", options=sorted(docs), placeholder="All documents"
        )
        for name, n in sorted(docs.items()):
            c1, c2 = st.columns([5, 1])
            c1.caption(f"📄 {name} ({n} chunks)")
            if c2.button("🗑️", key=f"del_{name}", help=f"Remove {name}"):
                rc.delete_document(vs, name)
                st.rerun()
    else:
        st.info("No documents yet. Upload PDFs above or put them in ./data and sync.")

    st.divider()
    k = st.slider("Passages to retrieve (k)", 2, 10, rc.TOP_K)
    if st.button("🧹 Clear chat"):
        ss.messages = []
        st.rerun()

# --------------------------------------------------------------------- chat
st.title("🎓 University AI Assistant")
st.caption("Ask questions based on university documents")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if question := st.chat_input("Ask your university question..."):
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not docs:
            reply, cards = "Please upload at least one PDF first.", []
            st.markdown(reply)
        else:
            with st.spinner("Searching documents..."):
                standalone, results = rc.ask(vs, question, history, selected or None, k)
            if standalone != question:
                st.caption(f"🔎 Searching for: *{standalone}*")
            reply = st.write_stream(rc.answer_stream(standalone, results))
            cards = rc.source_cards(results)
            if cards and rc.NOT_FOUND not in reply:
                show_sources(cards)
            else:
                cards = []
    ss.messages.append({"role": "assistant", "content": reply, "sources": cards})
