import streamlit as st
from rag_pipeline import get_summary, run_rag_pipeline, process_pdf

st.set_page_config(page_title="GenAI RAG Pipeline", page_icon="🔍", layout="centered")

st.title("🔍 GenAI RAG Pipeline")
st.write("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = process_pdf("uploaded.pdf")
    text = " ".join([d.page_content for d in docs])

    st.subheader("📄 Summary (under 200 words):")
    summary = get_summary(text)
    st.success(summary)

    st.subheader("❓ Ask a Question:")
    user_question = st.text_input("Enter your question about the document")

    if st.button("Run Query") and user_question:
        answer = run_rag_pipeline(docs, user_question)
        st.info(answer)
