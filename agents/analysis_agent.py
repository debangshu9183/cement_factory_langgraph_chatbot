from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def analysis_agent(state):

    context = state.get("merged_context", "")
    question = state.get("question", "")

    prompt = f"""You are an expert document analyst. Answer the question thoroughly and in detail.
Use specific examples, facts, and details from the context provided.
Do not summarize briefly — provide a comprehensive, well-structured answer.

Context:
{state['merged_context']}

Question: {state['question']}

Provide a detailed answer with:
- Key findings
- Specific details from the documents
- Examples where relevant
"""

    answer = llm.invoke(prompt).content

    return {"answer": answer}