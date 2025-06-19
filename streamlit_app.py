import os
import re
import json
import streamlit as st
from google.cloud import bigquery
from langchain_community.agents import create_react_agent, Tool
from langchain_community.agents.agent_types import AgentType
from langchain_huggingface import HuggingFaceEndpoint

# --------------------------------------------------
# 0. Page config
# --------------------------------------------------
st.set_page_config(page_title="OMOP SynPUF Text‑to‑SQL", layout="wide")

# --------------------------------------------------
# 1. Load secrets (fail fast if missing)
# --------------------------------------------------
if "GCP" not in st.secrets:
    st.error("❌ GCP credentials missing in secrets.toml.")
    st.stop()
if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets or not st.secrets["HUGGINGFACEHUB_API_TOKEN"]:
    st.error("❌ HUGGINGFACEHUB_API_TOKEN missing in secrets.toml.")
    st.stop()

# Build BigQuery client from service‑account info
sa_info = dict(st.secrets["GCP"])  # st.secrets behaves like a dict
bq_client = bigquery.Client.from_service_account_info(sa_info)

PROJECT = sa_info["project_id"]             # fluid-catfish-456819-v2
DATASET = "synpuf"                           # adjust if different
BQ_PATH = f"{PROJECT}.{DATASET}"

# --------------------------------------------------
# 2. Cache table list
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def list_tables():
    return sorted(tbl.table_id for tbl in bq_client.list_tables(BQ_PATH))

tables = list_tables()

# --------------------------------------------------
# 3. UI – table selector
# --------------------------------------------------
st.sidebar.header("Choose OMOP table")
active_table = st.sidebar.selectbox("Table", tables)

# --------------------------------------------------
# 4. Helpers
# --------------------------------------------------
def run_bigquery(sql: str):
    job = bq_client.query(sql, location="US")
    return [dict(r) for r in job.result()]

def _sanitize(name: str) -> str:
    return re.sub(r"[`\s\'\"]+", "", name)

# --------------------------------------------------
# 5. Tools
# --------------------------------------------------
def sql_tool(sql: str) -> str:
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

def describe_table(table: str) -> str:
    table = _sanitize(table)
    if "." in table:
        return "❌ Provide only the table name (e.g. 'person')."
    try:
        schema = bq_client.get_table(f"{BQ_PATH}.{table}").schema
        return "Schema:\n" + "\n".join(f"{f.name} ({f.field_type})" for f in schema)
    except Exception as exc:
        return f"Error: {exc}"

bigquery_tool = Tool("bigquery_query", sql_tool, "Run SQL on the chosen table.")
schema_tool   = Tool("describe_table", describe_table, "Show a table's columns.")

# --------------------------------------------------
# 6. LLM (Hugging Face Inference endpoint)
# --------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
    temperature=0.3,
    max_new_tokens=512,
)

# --------------------------------------------------
# 7. Agent
# --------------------------------------------------
system_msg = (
    f"You are a BigQuery SQL assistant. Active table: `{BQ_PATH}.{active_table}`. "
    "Use fully‑qualified names. Call `describe_table` if unsure about columns. "
    "Don't add LIMIT after aggregations."
)

agent = create_react_agent(
    llm=llm,
    tools=[bigquery_tool, schema_tool],
    system_message=system_msg,
    agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
)

# --------------------------------------------------
# 8. Main UI
# --------------------------------------------------
st.title("🩺 OMOP SynPUF Text‑to‑SQL (BigQuery)")
st.markdown(f"**Project**: `{PROJECT}` · **Dataset**: `{DATASET}` · **Table**: `{active_table}`")

question = st.text_area("Ask in natural language or SQL:")

if st.button("Run") and question.strip():
    with st.spinner("Running…"):
        try:
            answer = agent.run(question)
            st.success("Answer")
            st.code(answer)
        except Exception as exc:
            st.error(f"Agent error: {exc}")
