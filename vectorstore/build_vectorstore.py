from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def _make_embeddings():
    """Always creates a fresh instance — required for thread safety."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64}
    )

def build_vectorstore(docs, embeddings=None, force_rebuild: bool = False, index_path: str = "faiss_index"):

    #  Use provided embeddings or create a fresh one
    if embeddings is None:
        embeddings = _make_embeddings()

    #  Load from disk if it already exists
    if not force_rebuild and os.path.exists(index_path):
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)

    return db