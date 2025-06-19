# streamlit_app.py
import os
import re
import json
import streamlit as st
from google.cloud import bigquery

# LangChain & tools
from langchain_core.tools import Tool
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# --------------------------------------------------
# 0. Page config
# --------------------------------------------------
st.set_page_config(page_title="OMOP SynPUF Text‑to‑SQL", layout="wide")

# --------------------------------------------------
# 1. Secrets -> BigQuery client
# --------------------------------------------------
if "GCP" not in st.secrets:
    st.stop()

sa_info = dict(st.secrets["GCP"])
bq_client = bigquery.Client.from_service_account_info(sa_info)

PROJECT = sa_info["project_id"]           # e.g. fluid-catfish-456819-v2
DATASET = "synpuf"
BQ_PATH = f"{PROJECT}.{DATASET}"

# --------------------------------------------------
# 2. Table list (cache)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def list_tables():
    return sorted(t.table_id for t in bq_client.list_tables(BQ_PATH))

tables = list_tables()

# --------------------------------------------------
# 3. UI – pick table
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

bigquery_tool = Tool("bigquery_query", sql_tool_fn, "Run SQL on the chosen table.")
schema_tool   = Tool("describe_table", describe_table_fn, "Show a table's schema.")

tools = [bigquery_tool, schema_tool]
tool_names = ", ".join(t.name for t in tools)

# --------------------------------------------------
# 6. LLM (Hugging Face endpoint)
# --------------------------------------------------
if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
    st.error("Missing Hugging Face token in secrets.toml")
    st.stop()

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
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template=f"""
You are a BigQuery SQL assistant. Active table: `{BQ_PATH}.{active_table}`.
You have these tools:
{{tools}}

Use the format:
Question: {{input}}
Thought: ...
Action: one of [{{tool_names}}]
Action Input: ...
Observation: ...
... (repeat)
Thought: I have enough info
Final Answer: answer here

Begin!
Question: {{input}}
Thought:{{agent_scratchpad}}
"""
)

react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence="Observation:",
)

agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

# --------------------------------------------------
# 8. Main UI
# --------------------------------------------------
st.title("🩺 OMOP SynPUF Text‑to‑SQL")
st.markdown(f"**Project:** `{PROJECT}` · **Dataset:** `{DATASET}` · **Table:** `{active_table}`")

question = st.text_area("Ask anything (NL or SQL):")

if st.button("Run") and question.strip():
    with st.spinner("Running…"):
        try:
            result = agent_executor.invoke({"input": question})
            answer = result.get("output", "")
            st.success("Answer")
            st.code(answer)
        except Exception as exc:
            st.error(f"Agent error: {exc}")
