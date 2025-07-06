from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up LLM with valid model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"  # ✅ Supported model
)

# 📄 Load PDF and split it into pages
def process_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

# 📝 Summarize the full text (under 200 words)
def get_summary(text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following text in under 200 words."),
        ("human", "{text}")
    ])
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content

# 🤖 Run Retrieval-Augmented Generation (RAG) pipeline
def run_rag_pipeline(docs, query):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # ✅ Force CPU to fix meta tensor error
    )

    # Create FAISS vector store from PDF documents
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Build QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    response = qa.run(query)
    return response
