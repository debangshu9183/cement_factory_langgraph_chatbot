from typing import TypedDict

class GraphState(TypedDict):

    question: str
    query: str

    context_doc1: str
    context_doc2: str

    merged_context: str

    answer: str
    confidence: float