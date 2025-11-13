import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI 
# from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from typing_extensions import Annotated
from langgraph.graph import START, END, StateGraph

 
_ = load_dotenv(find_dotenv())
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY_US')
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2024-08-01-preview'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://azure-chat-try-2.openai.azure.com/'

# LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_PROJECT'] = "text-to-sql"

# llm = ChatOpenAI(model="gpt-4o", temperature=0) 
llm = AzureChatOpenAI(
    api_key = OPENAI_API_KEY,  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment="chat-endpoint-us-gpt4o"
)


db = SQLDatabase.from_uri("sqlite:///../../db-prep/credit-risk.db", sample_rows_in_table_info=2)


class State(TypedDict):
    question: str
    query: str
    feedback: str
    errors: str
    result: str
    answer: str


write_query_instructions = """
You are a SQL expert with a strong attention to detail.

Given an input question, create a syntactically correct SQLite query to run to help find the answer. 

When generating the query:

Unless the user specifies in his question a specific number of examples they wish to obtain, limit your query to at most 3 results. 
For example, if the user asks for the top 5 results, you should NOT limit the query to 3 results.

You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

Only use the following tables:
{schema}

Question: {question}
"""

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""

    question = state["question"]
    schema = db.get_table_info()
    
    prompt = write_query_instructions.format(
        question = question,
        schema = schema        
    ) 

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}    
    # return {"query": "SELECT SUM(NACE) AS total_off_balance_exposure FROM transactions WHERE DATE LIKE '2023-09%' AND EXPOSURE_TYPE = 'RRE' LIMIT 3;"}


rewrite_query_instructions = """
You are a SQL expert with a strong attention to detail.

You are provided with a previously written {previous_query} which is not correct.   

Re-write this query to refelct precisely the input question and the database tables to be queried.

When re-writing the query consider also the summary of {errors} if provided

Your goal is to create a syntactically correct SQLite query to run to help find the answer. 

When generating the query:

Unless the user specifies in his question a specific number of examples they wish to obtain, limit your query to at most 3 results. 
For example, if the user asks for the top 5 results, you should NOT limit the query to 3 results.

You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

Only use the following tables:
{schema}

Question: {question}
"""

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def rewrite_query(state: State):
    """Generate SQL query to fetch information."""

    previous_query = state['query']
    question = state["question"]
    schema = db.get_table_info()
    
    prompt = write_query_instructions.format(
        previous_query = previous_query,
        question = question,
        schema = schema        
    ) 

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)    
    return {"query": result["query"]}


query_check_instructions = """
Double check the {query} to make sure it reflects the input {question}.

Based on your check, respond with only 'correct' or 'incorrect'.

Also, if the query is incorrect, summarize the errors in the query in a short text to be for furter query refinement. If the quesry is correct do not output anything for errors.
"""

class FeedbackOutput(TypedDict):
    """The feedback of the query correctness check"""
    feedback: Annotated[str, ..., "The feedback could be 'correct' or incorrect'."]
    errors: Annotated[str, ..., "The errors in the query."] 

def check_query(state: State):
    question = state["question"]
    query = state['query']
    # schema = db.get_table_info()  
    
    prompt = query_check_instructions.format(
        question=question,
        query=query,
        # schema=schema        
    ) 

    structured_llm = llm.with_structured_output(FeedbackOutput)
    result = structured_llm.invoke(prompt)
    return {"feedback": result["feedback"], "errors": result["errors"]}


def run_query(state: State):
    query = state['query']
    result = db.run_no_throw(query)
    return {"result": result}


answer_instructions = """
Given the {feedback} on the {query} correctenss, either answer the user {question} based on the corresponding {query} and {result}, or ask to re-write the question.
If you think it is more appropriate, return the answer in the form of a table.
"""

def generate_answer(state: State):

    question = state["question"]
    query = state["query"]
    result = state.get("result", None)
    feedback = state.get('feedback', None)

    prompt = answer_instructions.format(
        question=question,
        query=query,
        result=result,
        feedback=feedback
    )   
    
    if feedback.lower() == 'correct':
        response = llm.invoke(prompt) 
        return {"answer": response.content}     
    else:        
        return {"answer": 'Please re-write your question! Be more specific and provide details about what you are looking for!'}


def regenarate_router(state: State): 
    if state['feedback'].lower() == 'correct':
        return 'run_query'        
    else:        
        return 'rewrite_query'


def answer_router(state: State): 
    if state['feedback'].lower() == 'correct':
        return 'run_query'        
    else:        
        return 'generate_answer'


builder = StateGraph(State)

builder.add_node('write_query', write_query)
builder.add_node('check_query', check_query)
builder.add_node('recheck_query', check_query)
builder.add_node('run_query', run_query)
builder.add_node('rewrite_query', rewrite_query)
builder.add_node('generate_answer', generate_answer)

builder.add_edge(START, 'write_query')
builder.add_edge('write_query', 'check_query')
builder.add_conditional_edges('check_query', regenarate_router, ['run_query', 'rewrite_query'])
builder.add_edge('rewrite_query', 'recheck_query')
builder.add_conditional_edges('recheck_query', answer_router, ['run_query', 'generate_answer'])
builder.add_edge('run_query', 'generate_answer') 
builder.add_edge('generate_answer', END)

graph = builder.compile()   