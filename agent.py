from typing import TypedDict, List, Optional
from typing_extensions import Annotated
import operator
from langgraph.graph import StateGraph, START, END

import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# ─── LLM Setup ────────────────────────────────────────────────────────────────

llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# ─── State Definition ─────────────────────────────────────────────────────────

class vehRegState(TypedDict):
    # Conversation
    message: Annotated[list, operator.add]
    previous_query: List[str]

    # Query
    user_query: str
    query_intent: str

    # Filtering
    filters: dict
    time_range: tuple

    # Data
    data_loaded: bool
    current_dataframe: pd.DataFrame
    dataset_registry: dict

    # Function tracking
    function_called: str

    # Human-in-the-loop
    human_approved_code_status: str
    human_feedback: Annotated[list[str], operator.add]

    # Code
    generated_code: str

    # Outputs
    analysis_result: dict
    summary: str
    error_log: List[str]


# ─── Structured Output Schema ──────────────────────────────────────────────────

class queryMetadata(BaseModel):
    intent: str = Field(description="Intent of the user query: trend_analysis, comparison, or breakdown.")
    states: List[str] = Field(description="States mentioned in the query.")
    fuel_type: List[str] = Field(description="Fuel types mentioned.")
    vehicle_category: Optional[str] = None
    vehicle_class: Optional[str] = None
    time_range: List[int]


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

def load_data(state: vehRegState):
    """Load preprocessed master CSV, generate if missing."""
    csv_path = "master_vehicle_data.csv"
    if not os.path.exists(csv_path):
        from pre_process import process_vehicle_data
        process_vehicle_data()
    df = pd.read_csv(csv_path)
    return {
        "current_dataframe": df,
        "data_loaded": True
    }


def extract_query_metadata(state: vehRegState):
    """Extract structured metadata from the user's natural language query."""
    query = state["user_query"]
    prompt = f'''
    Extract the following from the query:
    1. intent (trend_analysis, comparison, or breakdown — return ONLY the intent name)
    2. states
    3. fuel type
    4. vehicle category
    5. vehicle class
    6. time range

    Return JSON only and nothing else.

    Query: {query}
    '''
    structured_llm = llm.with_structured_output(queryMetadata)
    parsed = structured_llm.invoke(prompt)

    return {
        "query_intent": parsed.intent,
        "filters": {
            "state": parsed.states,
            "fuel_type": parsed.fuel_type,
            "vehicle_category": parsed.vehicle_category,
            "vehicle_class": parsed.vehicle_class
        },
        "time_range": parsed.time_range,
    }


def route_analysis(state: vehRegState):
    """Routing function — decides which analysis node to call next."""
    intent = state["query_intent"]
    if intent == "trend_analysis":
        return "trend_function"
    elif intent == "comparison":
        return "comparison_function"
    elif intent == "breakdown":
        return "breakdown_function"
    return "trend_function"  # fallback


def clean_code(code: str) -> str:
    """Strip markdown code fences from LLM output."""
    code = code.replace("```python", "").replace("```", "")
    return code.strip()


def trend_function(state: vehRegState):
    """Generate Python code for trend analysis."""
    df = state["current_dataframe"]
    schema_info = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    feedback = state.get("human_feedback", [])
    sample_rows = df.head(4).to_dict()

    prompt = f"""
    You are a pandas data analysis assistant.
    A pandas dataframe named `df` already exists.

    DO NOT load CSV files. DO NOT use pd.read_csv(). Use the existing `df`.

    Dataset schema: {schema_info}
    Sample rows: {sample_rows}
    User Query: "{state['user_query']}"
    Human Feedback (if any): {feedback}

    Generate ONLY executable Python code that:
    - Uses the existing `df`
    - Performs trend analysis
    - Uses matplotlib for plots (do NOT call plt.show())
    - Saves the plot as 'chart.png' using plt.savefig('chart.png', bbox_inches='tight')
    - Stores final result in variable `result`

    Return ONLY Python code.
    """
    code = llm.invoke(prompt).content
    return {
        "generated_code": clean_code(code),
        "function_called": "trend_function"
    }


def comparison_function(state: vehRegState):
    """Generate Python code for comparison analysis."""
    df = state["current_dataframe"]
    schema_info = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    feedback = state.get("human_feedback", [])
    sample_rows = df.head(4).to_dict()

    prompt = f"""You are a pandas data analysis expert specializing in comparative analysis.
    A DataFrame named `df` is already loaded in memory.

    Dataset Info:
    Columns: {schema_info['columns']}
    Sample: {sample_rows}
    User Query: "{state['user_query']}"
    Human Feedback (if any): {feedback}

    Requirements:
    - Use the existing `df` (DO NOT load any CSV)
    - Compare 2 or more entities (states, fuel types, categories, etc.)
    - Create a chart using matplotlib (do NOT call plt.show())
    - Save plot as 'chart.png' using plt.savefig('chart.png', bbox_inches='tight')
    - Store results in variable `result` as a dictionary or DataFrame

    Return ONLY executable Python code. No explanations."""

    code = llm.invoke(prompt).content
    return {
        "generated_code": clean_code(code),
        "function_called": "comparison_function"
    }


def breakdown_function(state: vehRegState):
    """Generate Python code for breakdown/distribution analysis."""
    df = state["current_dataframe"]
    schema_info = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    feedback = state.get("human_feedback", [])
    sample_rows = df.head(3).to_dict()

    prompt = f"""You are a pandas data analysis expert specializing in distribution analysis.
    A DataFrame named `df` is already loaded in memory.

    Dataset Info:
    Columns: {schema_info['columns']}
    Sample: {sample_rows}
    User Query: "{state['user_query']}"
    Human Feedback (if any): {feedback}

    Requirements:
    - Use the existing `df` (DO NOT load any CSV)
    - Show composition/distribution of a category
    - Calculate percentage or count for each category
    - Create a pie chart or horizontal bar chart using matplotlib (do NOT call plt.show())
    - Save plot as 'chart.png' using plt.savefig('chart.png', bbox_inches='tight')
    - Store results in variable `result` as a dictionary with percentages

    Return ONLY executable Python code. No explanations."""

    code = llm.invoke(prompt).content
    return {
        "generated_code": clean_code(code),
        "function_called": "breakdown_function"
    }

# STREAMLIT NOTE: The human_approval node is bypassed in the Streamlit app, as approval is 
# handled directly in app.py via session state. 
# This function exists only to maintain graph structure if run standalone.

def human_approval(state: vehRegState):
    """
    REPLACED by Streamlit UI — this node is bypassed in the Streamlit app.
    The approval is handled directly in app.py via session state.
    This stub exists only to keep the graph structure intact if run standalone.
    """
    return {
        "human_approved_code_status": "yes",
        "human_feedback": []
    }


def approval_router(state: vehRegState):
    """Route based on human approval status."""
    status = state["human_approved_code_status"]
    if status.lower() in ["yes", "ok", "approved", "approve"]:
        return "code_viewer"
    return state["function_called"]


def code_viewer(state: vehRegState):
    """Execute the LLM-generated code safely."""
    code = state["generated_code"]
    df = state["current_dataframe"]

    local_namespace = {"df": df, "pd": pd, "np": np, "plt": plt}

    try:
        exec(code, local_namespace)
        result = local_namespace.get("result")
        return {
            "analysis_result": result,
            "error_log": []
        }
    except Exception as e:
        return {
            "analysis_result": None,
            "error_log": [str(e)]
        }


def summarize(state: vehRegState):
    """Generate a human-friendly summary of the analysis result."""
    prompt = f"""
    You are a professional vehicle registration data analyst.
    Explain the analytical results in a clear, concise, human-friendly manner.

    The user originally asked: "{state['user_query']}"
    The analysis result is: {state['analysis_result']}

    Instructions:
    - Answer the user's question directly.
    - Explain the trend/pattern clearly with key numbers.
    - Highlight increases, decreases, peaks, or anomalies.
    - DO NOT mention pandas, Python, dataframe, or code execution.
    - Keep it professional but conversational.

    Return ONLY the final explanation.
    """
    summary = llm.invoke(prompt).content
    return {"summary": summary}


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_workflow():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(vehRegState)

    graph.add_node("load_data", load_data)
    graph.add_node("extract_query_metadata", extract_query_metadata)
    graph.add_node("trend_function", trend_function)
    graph.add_node("comparison_function", comparison_function)
    graph.add_node("breakdown_function", breakdown_function)
    graph.add_node("human_approval", human_approval)
    graph.add_node("code_viewer", code_viewer)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "extract_query_metadata")

    graph.add_conditional_edges(
        "extract_query_metadata",
        route_analysis,
        {
            "trend_function": "trend_function",
            "comparison_function": "comparison_function",
            "breakdown_function": "breakdown_function"
        }
    )

    graph.add_edge("trend_function", "human_approval")
    graph.add_edge("comparison_function", "human_approval")
    graph.add_edge("breakdown_function", "human_approval")

    graph.add_conditional_edges(
        "human_approval",
        approval_router,
        {
            "code_viewer": "code_viewer",
            "trend_function": "trend_function",
            "comparison_function": "comparison_function",
            "breakdown_function": "breakdown_function"
        }
    )

    graph.add_edge("code_viewer", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
