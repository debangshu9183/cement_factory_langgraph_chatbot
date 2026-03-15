from langchain_community.document_loaders import PyPDFLoader

def load_documents():

    loader1 = PyPDFLoader(r"F:\AI_PROJECTS\langgraph_chatbot\documents\doc1.pdf")
    loader2 = PyPDFLoader(r"F:\AI_PROJECTS\langgraph_chatbot\documents\doc2.pdf")

    doc1 = loader1.load()
    doc2 = loader2.load()

    return doc1, doc2

print(len(load_documents()[0]), len(load_documents()[1]))
