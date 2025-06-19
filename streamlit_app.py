# streamlit_app.py
import os
import re
import json
import streamlit as st
from google.cloud import bigquery

# LangChain imports (v0.2+)
from langchain_core.tools import Tool
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# --------------------------------------------------
# 0. Streamlit page config
# --------------------------------------------------
st.set_page_config(page_title="OMOP SynPUF Text‑to‑SQL", layout="wide")

# --------------------------------------------------
# 1. Secrets → BigQuery client
# --------------------------------------------------
if "GCP" not in st.secrets or "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
    st.error("Add GCP service‑account and Hugging Face token in secrets.toml.")
    st.stop()

sa_info = dict(st.secrets["GCP"])
bq_client = bigquery.Client.from_service_account_info(sa_info)

PROJECT = sa_info["project_id"]        # e.g. fluid-catfish-456819-v2
DATASET = "synpuf"
BQ_PATH = f"{PROJECT}.{DATASET}"

# --------------------------------------------------
# 2. List tables (cache)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def list_tables():
    return sorted(t.table_id for t in bq_client.list_tables(BQ_PATH))

tables = list_tables()

# --------------------------------------------------
# 3. Pick active table
# --------------------------------------------------
st.sidebar.header("Choose OMOP table")
active_table = st.sidebar.selectbox("Table", tables)

# --------------------------------------------------
# 4. BigQuery helpers
# --------------------------------------------------
def run_bigquery(sql: str):
    job = bq_client.query(sql, location="US")
    return [dict(r) for r in job.result()]

def _sanitize(name: str) -> str:
    return re.sub(r"[`\s\'\"]+", "", name)

# --------------------------------------------------
# 5. Tools
# --------------------------------------------------
def sql_tool_fn(sql: str) -> str:
    sql = sql.replace("\n", " ")
    if f"FROM {active_table}" in sql and BQ_PATH not in sql:
        sql = sql.replace(f"FROM {active_table}", f"FROM `{BQ_PATH}.{active_table}`")
    if BQ_PATH not in sql:
        return f"⚠️ Query must reference `{BQ_PATH}.{active_table}`."
    try:
        rows = run_bigquery(sql)
        return "Query returned no rows." if not rows else str(rows[:5])
    except Exception as exc:
        return f"Query error: {exc}"

def describe_table_fn(table: str) -> str:
    table = _sanitize(table)
    if "." in table:
        return "❌ Provide only the table name (e.g. 'person')."
    try:
        schema = bq_client.get_table(f"{BQ_PATH}.{table}").schema
        return "\n".join(f"{f.name} ({f.field_type})" for f in schema)
    except Exception as exc:
        return f"Error: {exc}"

bigquery_tool = Tool("bigquery_query", sql_tool_fn, "Run SQL on the active table.")
schema_tool   = Tool("describe_table", describe_table_fn, "Show a table's schema.")

tools = [bigquery_tool, schema_tool]
tool_names_str = ", ".join(t.name for t in tools)
tool_desc_str  = "\n".join(f"{t.name}: {t.description}" for t in tools)

# --------------------------------------------------
# 6. LLM (Hugging Face endpoint)
# --------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
    temperature=0.3,
    max_new_tokens=512,
)

# --------------------------------------------------
# 7. Prompt template & agent
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "table", "tool_descriptions", "tool_names"],
    template="""
You are a BigQuery SQL assistant. Active table: `{table}`.

You have access to these tools:
{tool_descriptions}

Use this interaction format:

Question: the question to answer
Thought: you should think what to do
Action: one of [{tool_names}]
Action Input: the input for that action
Observation: the result
... (repeat Thought / Action / Action Input / Observation)
Thought: I now know the final answer
Final Answer: the answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
"""
)

react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt.partial(
        table=f"{BQ_PATH}.{active_table}",
        tool_descriptions=tool_desc_str,
        tool_names=tool_names_str,
    ),
    stop_sequence="Observation:",
)

agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

# --------------------------------------------------
# 8. Streamlit UI
# --------------------------------------------------
st.title("🩺 OMOP SynPUF Text‑to‑SQL")
st.markdown(
    f"**Project**: `{PROJECT}` · **Dataset**: `{DATASET}` · "
    f"**Table**: `{active_table}`"
)

query = st.text_area("Ask a question (natural language or raw SQL):")

if st.button("Run") and query.strip():
    with st.spinner("Running…"):
        try:
            result = agent_executor.invoke({"input": query})
            answer = result.get("output", "")
            st.success("Answer")
            st.code(answer)
        except Exception as exc:
            st.error(f"Agent error: {exc}")
