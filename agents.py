from typing import TypedDict
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode
from langchain_core.tools import Tool
from langchain_core.messages.base import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from vectordb import retriever
from dotenv import load_dotenv
from graders import check_relevance , check_halluc , check_ans
from llm_utils import run_fallback_chain, question_router
from llm_utils import rag_chain
from data.data_URL import ABAP_TOPICS
from typing import Union

load_dotenv()

tavily_search = TavilySearchResults()
tool_executor = ToolNode(
    tools=[
        Tool(
            name="VectorStore",
            func=retriever.invoke,
            description="Useful to search the vector database",
        ),
        Tool(
            name="SearchEngine", 
            func=tavily_search, 
            description="Useful to search the web"
        ),
    ]
)


class AgentState(TypedDict):
    """The dictionary keeps track of the data required by the various nodes in the graph"""

    query: str
    chat_history:list[BaseMessage]
    generation: str
    documents: list[Document]


def retrieve_node(state: dict) -> dict[str, Union[list[Document], str]]:
    """
    Retrieve relevent documents from the vectorstore
        query: str

    return list[Document]
    """
    query = state["query"]
    documents = retriever.invoke(input=query)
    return {"documents": documents}


def fallback_node(state: dict):
    """
    Fallback to this node when there is no tool call
    """
    query = state["query"]
    chat_history = state["chat_history"]
    generation = run_fallback_chain(query,chat_history)
    return {"generation": generation}


def filter_documents_node(state: dict):
    filtered_docs = list()

    query = state["query"]
    documents = state["documents"]
    for i, doc in enumerate(documents, start=1):
        grade = check_relevance(doc, query)
        if grade == "relevant":
            print(f"---CHUCK {i}: RELEVANT---")
            filtered_docs.append(doc)
        else:
            print(f"---CHUCK {i}: NOT RELEVANT---")
    return {"documents": filtered_docs}


def rag_node(state: dict):
    query = state["query"]
    documents = state["documents"]

    generation = rag_chain.invoke({"query" : query,"context" : documents})
    return {"generation": generation}


# def web_search_node(state: dict):
#     query = state["query"]
#     results = tavily_search.invoke(query)
#     documents = [
#         Document(page_content=doc["content"], metadata={"source": doc["url"]})
#         for doc in results
#     ]
#     print(documents)
#     return {"documents": documents}
def web_search_node(state: dict):
    query = state["query"]
    results = tavily_search.invoke(query)
    
    # Debug: Print the raw results before processing
    print(f" Raw results from tavily_search: {type(results)} - {results}")

    # Check and print each item in results
    for idx, doc in enumerate(results):
        print(f"🔍 [DEBUG] Item {idx}: Type: {type(doc)}, Value: {doc}")

    # Processing results into Document objects
    try:
        documents = [
            Document(page_content=doc["content"], metadata={"source": doc["url"]})
            for doc in results
        ]
    except Exception as e:
        print(f"Exception while creating Document objects: {e}")
        return {"documents": []}  # Return empty list in case of error

    print(f" Processed Documents: {documents}")
    return {"documents": documents}



def question_router_node(state: dict):
    query = state["query"]
    topics_list = ", ".join(ABAP_TOPICS)
    try:
        response = question_router.invoke({"query" : query , "topics_list" : topics_list})
        print(f"Router Response: {response}")
    except Exception as e:
        print(f" Router failed: {str(e)}")
        return "llm_fallback"

    if "tool_calls" not in response.additional_kwargs:
        print("---No tool called---")
        return "llm_fallback"

    if len(response.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide route!"

    route = response.additional_kwargs["tool_calls"][0]["function"]["name"]
    print(f" Routing to: {route}")
    if route == "VectorStore":
        print("---Routing to VectorStore---")
        return "VectorStore"
    elif route == "SearchEngine":
        print("---Routing to SearchEngine---")
        return "SearchEngine"
    print(" Unknown route, defaulting to fallback.")
    return "llm_fallback"


def should_generate(state: dict):
    filtered_docs = state["documents"]
    if not filtered_docs:
        print("---All retrieved documents not relevant---")
        return "SearchEngine"
    else:
        print("---Some retrieved documents are relevant---")
        return "generate"


def hallucination_and_answer_relevance_check(state: dict):
    llm_response = state["generation"]
    documents = state["documents"]
    query = state["query"]

    hallucination_grade = check_halluc(
        documents, llm_response
    )
    if hallucination_grade == "no":
        print("---Hallucination check passed---")
        answer_relevance_grade = check_ans(
            query, llm_response
        )
        if answer_relevance_grade == "yes":
            print("---Answer is relevant to question---\n")
            return "useful"
        else:
            print("---Answer is not relevant to question---")
            return "not useful"
    print("---Hallucination check failed---")
    return "generate"

