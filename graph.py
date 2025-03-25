from langgraph.graph import StateGraph, END
from agents import retrieve_node , web_search_node , filter_documents_node, fallback_node , rag_node, question_router_node , AgentState , should_generate , hallucination_and_answer_relevance_check
from io import BytesIO
from PIL import Image
from IPython.display import display

workflow = StateGraph(AgentState)
workflow.add_node("VectorStore", retrieve_node)
workflow.add_node("SearchEngine", web_search_node)
workflow.add_node("filter_docs", filter_documents_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("rag", rag_node)

workflow.set_conditional_entry_point(
    question_router_node,
    {
        "llm_fallback": "fallback",
        "VectorStore": "VectorStore",
        "SearchEngine": "SearchEngine",
    },
)

workflow.add_edge("VectorStore", "filter_docs")
workflow.add_edge("SearchEngine", "filter_docs")
workflow.add_conditional_edges(
    "filter_docs", should_generate, {"SearchEngine": "SearchEngine", "generate": "rag"}
)
workflow.add_conditional_edges(
    "rag",
    hallucination_and_answer_relevance_check,
    {"useful": END, "not useful": "SearchEngine", "generate": "rag"},
)

workflow.add_edge("fallback", END)


app = workflow.compile(debug=True)



