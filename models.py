from pydantic import BaseModel , Field
from data.data_URL import ABAP_TOPICS


class VectorStore(BaseModel):
    """A vector store containing information on RAP, CDS, Regex, RFC, SQL, Selection, Transaction, Buffering, Update, Expressions, Transformation, and XML."""
    query: str = Field(..., description="ABAP-related query for retrieval from the VectorStore.")

class SearchEngine(BaseModel):
    """A search engine for searching other ABAP-related information that is not stored in the VectorStore."""
    query: str = Field(..., description="ABAP-related query for external lookup.")


def get_router_prompt(query: str) -> str:
    """Returns a prompt for routing the user query."""
    topics_list = ", ".join(ABAP_TOPICS)
    return (
        "You are an expert in routing user queries to either a VectorStore or SearchEngine.\n"
        "Use SearchEngine for all other ABAP related queries that are not related to ABAP topics in vector store\n"
        "The VectorStore contains information on the following topics :\n"
        f"topics list : {topics_list}"
        'If a query is not ABAP related, output "not ABAP-related" without using any tool.\n\n'
        f"query: {query}"
    )