import streamlit as st
import os
from google.cloud import bigquery
from langchain.llms import Ollama
from langchain.agents import Tool, AgentType, initialize_agent

# Set Streamlit page config FIRST
st.set_page_config(page_title="OMOP SynPUF Text-to-SQL on BigQuery", layout="wide")

# --- GCP credentials ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\TNEL\Downloads\fluid-catfish-456819-v2-a835705834b7.json"

# --- Initialize BigQuery Client ---
client = bigquery.Client()
PROJECT = "fluid-catfish-456819-v2"
DATASET = "synpuf"  # e.g. "hhs_synpuf"
FULL_DATASET = f"{PROJECT}.{DATASET}"

# --- List all tables in the dataset ---
@st.cache_data
def list_tables():
    tables = client.list_tables(f"{FULL_DATASET}")
    return [table.table_id for table in tables]

available_tables = list_tables()

# --- Let user choose a table ---
selected_table = st.selectbox("Choose a table from the SynPUF dataset:", available_tables)

# --- Run BigQuery with selected table in scope ---
def run_bigquery(query: str):
    job = client.query(query, location="US")
    return [dict(row) for row in job.result()]

# --- LangChain Tool wrapping BigQuery ---
def bigquery_tool_fn(query: str) -> str:
    if selected_table not in query:
        return f"Error: Query must include the selected table `{selected_table}`"
    try:
        rows = run_bigquery(query)
        if not rows:
            return "Query returned no results."
        return str(rows[:5])
    except Exception as e:
        return f"Query error: {e}"

bigquery_tool = Tool(
    name="bigquery_query",
    func=bigquery_tool_fn,
    description=f"Run SQL queries against the `{selected_table}` table in the SynPUF dataset."
)

# --- Ollama LLM ---
llm = Ollama(model="llama3")

# --- Create LangChain agent ---
agent_executor = initialize_agent(
    
system_message = (
    "You are an SQL assistant for BigQuery. "
    "All tables live in dataset `synpuf` (location US) "
    "inside the current project. "
    "Always reference tables as `synpuf.table_name`."
    ),
    tools=[bigquery_tool],
    llm=llm,
    agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# --- Streamlit UI ---
st.title("🩺 OMOP SynPUF Text-to-SQL (BigQuery)")
user_query = st.text_area("Enter your natural language question (must use the selected table):")

if st.button("Run Query") and user_query.strip():
    with st.spinner("Thinking..."):
        try:
            result = agent_executor.run(user_query)
            st.subheader("💬 Response")
            st.text(result)
        except Exception as e:
            st.error(f"Agent error: {e}")
