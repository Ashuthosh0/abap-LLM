import os
from llm_config import llm 
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from vectordb import get_retriever
from langchain_core.messages import HumanMessage, AIMessage
from models import VectorStore, SearchEngine
from dotenv import load_dotenv
from data.data_URL import ABAP_TOPICS

load_dotenv()

retriever = get_retriever()
rag_template_str = (
    "You are a professional ABAP developer and expert ABAP coder. "
    "Optimize the given ABAP code to the fullest extent while maintaining functionality and best practices. "
    "Base your response strictly on the provided context. "
    "Display the source code from the context provided when the user asks for an example on a certain topic. "
    "DO NOT USE EXTERNAL ASSUMPTIONS ONLY USE CONTEXT PROVIDED. "
    "DO NOT RESPOND TO QUERIES NOT RELATED TO CONTEXT. "
    "Do not introduce any external assumptions. Do not use your personal knowledge.\n\n"
    "Context: {context}\n\n"
    "Query: {query}"
)

fallback_prompt = ChatPromptTemplate.from_template(
    (
        "You are an SAP assistant specializing in ABAP development.\n"
        "Only respond to queries related to SAP and ABAP.\n"
        "If a query is unrelated, politely acknowledge your limitation.\n"
        "Provide concise and precise responses for ABAP-related topics.\n\n"
        "Current conversation:\n\n{chat_history}\n\n"
        "Human: {query}"
    )
)


rag_prompt = ChatPromptTemplate.from_template(rag_template_str)
rag_chain = rag_prompt | llm | StrOutputParser()

def get_question_router():
    """Creates a question router using LLM and tools."""
    topics_list = ", ".join(ABAP_TOPICS)
    prompt_txt = (
        "You are an expert in routing user queries to either a VectorStore or SearchEngine.\n"
        "The VectorStore contains information on the following topics :\n"
        f"topics list : {topics_list}"
        "Use SearchEngine for all other ABAP-related queries that are not related to topics in the vector store.\n"
        "Tell the user which tool you used to answer the question (either the vector store or web search). "
        "If a query is not ABAP-related, output 'not ABAP' without using any tool.\n\n"
        "query: {query}"
    )
    return ChatPromptTemplate.from_template(prompt_txt) | llm.bind_tools(tools=[VectorStore, SearchEngine])

question_router = get_question_router()

def run_rag_chain(query: str, retriever) -> str:
    """Runs the RAG chain to generate an answer based on retrieved context."""
    context = retriever.invoke(query)
    return rag_chain.invoke({"query": query, "context": context})


fallback_chain = (
    {
        "chat_history": lambda x: "\n".join(
            [
                (
                    f"human: {msg.content}"
                    if isinstance(msg, HumanMessage)
                    else f"AI: {msg.content}"
                )
                for msg in x["chat_history"]
            ]
        ),
        "query": itemgetter("query"),
    }
    | fallback_prompt
    | llm
    | StrOutputParser()
)


def run_fallback_chain(query: str, chat_history=None) -> str:
    """Runs the fallback chain for Not-ABAP queries."""
    if chat_history is None:
        chat_history = []
    return fallback_chain.invoke({"query": query, "chat_history": chat_history})


