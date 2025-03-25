from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from typing import Literal
from llm_config import llm 


class Grader(BaseModel):
    """Use this format to grade relevance of retrieved documents."""

    grade: Literal["relevant", "irrelevant"] = Field(
        ..., description="The relevance score for the document.\n"
        "Set this to 'relevant' if the given context is relevant to the user's query, or 'irrelevant' if the document is not relevant.",
    )

    @validator("grade", pre=True)
    def validate_grade(cls, value):
        """Normalize 'not relevant' -> 'irrelevant'."""
        return "irrelevant" if value.lower() in ["not relevant", "irrelevant"] else "relevant"

# ✅ Relevance Grader Prompt
grader_system_prompt_template =  """You are a grader and an ABAP development expert tasked with assessing the relevance of a given context to a query.
    If the context is relevant to the query, score it as "relevant". Otherwise, give "irrelevant".
    Do not answer the actual answer, just provide the grade in JSON format with "grade" as the key, without any additional explanation."""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt_template),
        ("human", "context: {context}\n\nquery: {query}"),
    ]
)

# ✅ Define Grader Chain
grader_chain = grader_prompt | llm.with_structured_output(Grader, method="json_mode")

def check_relevance( documents: list, query: str) -> str:
    """
    Grades a list of retrieved documents for relevance to the query.
    """
    if not documents:
        print(" No docs")
        return "irrelevant"  
    
    # for doc in documents:
    #     response = grader_chain.invoke({"query": query, "context": doc.page_content})
    #     print(f" Document: {doc.page_content[:100]}... | Grade: {response.grade}") 
    for doc_tuple in documents:
        doc = doc_tuple[0]  # Extract the first element (actual document)
        response = grader_chain.invoke({"query": query, "context": doc})
        print(f" Document: {doc[:100]}... | Grade: {response.grade}") 

        if response.grade == "relevant":
            return "relevant"
    
    return "irrelevant"


    
    
class HallucinationGrader(BaseModel):
    "Binary score for hallucination check in llm's response"

    grade: Literal["yes", "no"] = Field(
        ..., description="'yes' if the llm's response is hallucinated otherwise 'no'"
    )


hallucination_grader_system_prompt_template = (
    "You are a grader and an ABAP expert assessing whether a response from an llm is based on a given context.\n"
    "If the llm's response is not based on the given context give a score of 'yes' meaning it's a hallucination"
    "otherwise give 'no'\n"
    "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
)

hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_system_prompt_template),
        ("human", "context: {context}\n\nllm's response: {response}"),
    ]
)


hallucination_grader_chain = (
    RunnableParallel(
        {
            "response": itemgetter("response"),
            # "context": lambda x: "\n\n".join([c.page_content for c in x["context"]]),
            "context": lambda x: "\n\n".join(x["context"]) if isinstance(x["context"], list) else x["context"],
        }
    )
    | hallucination_grader_prompt
    | llm.with_structured_output(HallucinationGrader, method="json_mode")
)

def check_halluc(documents: list, llm_res: str) -> str:
    """
    Checks if the LLM's response is hallucinated.
    Returns 'yes' if hallucinated, otherwise 'no'.
    """
    # Extract document contents
    # all_docs = [doc.page_content for doc in documents]
    all_docs = lambda x:"\n\n".join([doc.page_content for doc in x["documents"]])
    
    # Run hallucination grader
    res = hallucination_grader_chain.invoke({ "response": llm_res, "context": all_docs})
    print(f"🏷 Hallucination Grade: {res.grade}")  # Debugging print
    return res.grade  # Returns "yes" (hallucinated) or "no" (not hallucinated)


    
class AnswerGrader(BaseModel):
    "Binary score for an answer check based on a query."

    grade: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if the provided answer is an actual answer to the query otherwise 'no'",
    )


answer_grader_system_prompt_template = (
    "You are a grader assessing whether a provided answer is in fact an answer to the given query.\n"
    "If the provided answer does not answer the query give a score of 'no' otherwise give 'yes'\n"
    "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
)

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system_prompt_template),
        ("human", "query: {query}\n\nanswer: {response}"),
    ]
)


answer_grader_chain = answer_grader_prompt | llm.with_structured_output(
    AnswerGrader, method="json_mode"
)

def check_ans(query : str , llm_res : str) -> str:
    res = answer_grader_chain.invoke({"query" : query , "response" : llm_res})

    return res.grade  
