from langgraph.graph import StateGraph, END
from state import GraphState

from agents.query_analyzer import query_analyzer
from agents.query_rewriter import rewrite_query
from agents.analysis_agent import analysis_agent
from agents.confidence_checker import confidence_checker
from retrievers.retriever import fetch_doc1, fetch_doc2
from retrievers.retriever import merge_contexts          # add this here too

workflow = StateGraph(GraphState)

workflow.add_node("query_analyzer", query_analyzer)
workflow.add_node("fetch_doc1", fetch_doc1)
workflow.add_node("fetch_doc2", fetch_doc2)
workflow.add_node("merge_contexts", merge_contexts)
workflow.add_node("analysis_agent", analysis_agent)
workflow.add_node("confidence_check", confidence_checker)
workflow.add_node("rewrite_query", rewrite_query)

workflow.set_entry_point("query_analyzer")

# Fan out to both fetchers simultaneously
workflow.add_edge("query_analyzer", "fetch_doc1")
workflow.add_edge("query_analyzer", "fetch_doc2")

# Fan back in — LangGraph waits for both before proceeding
workflow.add_edge("fetch_doc1", "merge_contexts")
workflow.add_edge("fetch_doc2", "merge_contexts")

workflow.add_edge("merge_contexts", "analysis_agent")
workflow.add_edge("analysis_agent", "confidence_check")

def decision(state):
    return "rewrite_query" if state["confidence"] < 0.5 else END

workflow.add_conditional_edges("confidence_check", decision)
workflow.add_edge("rewrite_query", "analysis_agent")

app = workflow.compile()