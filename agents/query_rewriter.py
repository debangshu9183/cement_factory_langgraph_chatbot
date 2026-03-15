from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
def rewrite_query(state):

    query = state["query"]

    prompt = f"""
Rewrite the following query to improve document retrieval.

Query:
{query}
"""

    new_query = llm.invoke(prompt).content

    return {"query": new_query}