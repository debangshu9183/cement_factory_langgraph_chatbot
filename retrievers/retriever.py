from vectorstore.build_vectorstore import build_vectorstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# ✅ Separate embedding instance per retriever — fixes "Already borrowed" crash
def _make_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64}
    )

def _load_retriever(pdf_path: str, k: int = 5):
    index_path = pdf_path.replace(".pdf", "_index")
    embeddings = _make_embeddings()   # ✅ fresh instance each time

    if os.path.exists(index_path):
        print(f"Loading cached index: {index_path}")
        db = FAISS.load_local(index_path, embeddings,
                              allow_dangerous_deserialization=True)
    else:
        print(f"Building index for: {pdf_path}")
        docs = PyPDFLoader(pdf_path).load()
        db = build_vectorstore(docs, embeddings)
        db.save_local(index_path)

    return db.as_retriever(search_kwargs={"k": k})


_retriever_doc1 = _load_retriever(r"F:\AI_PROJECTS\langgraph_chatbot\documents\doc1.pdf")
_retriever_doc2 = _load_retriever(r"F:\AI_PROJECTS\langgraph_chatbot\documents\doc2.pdf")


def retrieve_context(query, retriever):
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

def fetch_doc1(state: dict) -> dict:
    query = state.get("query") or state.get("question")
    return {"context_doc1": retrieve_context(query, _retriever_doc1)}

def fetch_doc2(state: dict) -> dict:
    query = state.get("query") or state.get("question")
    return {"context_doc2": retrieve_context(query, _retriever_doc2)}

def merge_contexts(state: dict) -> dict:
    merged = f"{state.get('context_doc1', '')}\n\n{state.get('context_doc2', '')}"
    return {"merged_context": merged.strip()}