# streamlit_app.py
import os
import re
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# ------------------------
# 0. Streamlit page config
# ------------------------
st.set_page_config(
    page_title="OMOP SynPUF Text-to-SQL on BigQuery",
    layout="wide",
)

# ------------------------
# 1. BigQuery credentials & constants  (edit your secret path)
# ------------------------
# For Streamlit Cloud, set GOOGLE_APPLICATION_CREDENTIALS via secrets or config
# Example: os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/creds.json"
# Here we assume credentials are already configured on Streamlit Cloud environment

PROJECT = "fluid-catfish-456819-v2"
DATASET = "synpuf"
BQ_PATH = f"{PROJECT}.{DATASET}"

gcp_info = st.secrets["gcp"]

credentials = service_account.Credentials.from_service_account_info(gcp_info)
project_id = gcp_info["project_id"]

client = bigquery.Client(credentials=credentials, project=project_id)

# ------------------------
# 2. Cached helper to list tables
# ------------------------
@st.cache_data(show_spinner=False)
def list_tables():
    return sorted(tbl.table_id for tbl in client.list_tables(BQ_PATH))

tables = list_tables()

# ------------------------
# 3. Sidebar: choose active table
# ------------------------
st.sidebar.header("Choose OMOP table")
active_table = st.sidebar.selectbox("Table", tables)

# ------------------------
# 4. Helper to run BigQuery SQL
# ------------------------
def run_bigquery(sql: str):
    job = client.query(sql, location="US")
    return [dict(row) for row in job.result()]

# ------------------------
# 5. Sanitize utility for table names
# ------------------------
def _sanitize(name: str) -> str:
    return re.sub(r"[`\s\'\"]+", "", name)

# ------------------------
# 6. Tool A: SQL query tool
# ------------------------
def sql_tool(sql: str) -> str:
    sql = sql.replace("\n", " ")  # flatten newlines
    if f"FROM {active_table}" in sql and BQ_PATH not in sql:
        sql = sql.replace(f"FROM {active_table}", f"FROM `{BQ_PATH}.{active_table}`")
    if BQ_PATH not in sql:
        return f"⚠️ Query must reference `{BQ_PATH}.{active_table}`."
    try:
        rows = run_bigquery(sql)
        if not rows:
            return "Query returned no rows."
        return str(rows[:5])
    except Exception as e:
        return f"Query error: {e}"

bigquery_tool = Tool(
    name="bigquery_query",
    func=sql_tool,
    description=f"Run StandardSQL against `{BQ_PATH}.{active_table}` and return the first 5 rows.",
)

# ------------------------
# 7. Tool B: describe table schema
# ------------------------
def describe_table(table: str) -> str:
    table = _sanitize(table)
    if "." in table:
        return "❌ Pass only the table name, e.g. 'person'."
    try:
        tbl = client.get_table(f"{BQ_PATH}.{table}")
        lines = [f"{field.name} ({field.field_type})" for field in tbl.schema]
        return f"Schema for `{table}`:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error describing table `{table}`: {e}"

schema_tool = Tool(
    name="describe_table",
    func=describe_table,
    description="Return columns & types for a table (e.g. 'person').",
)

# ------------------------
# 8. LLM and prompt setup
# ------------------------
llm = HuggingFaceEndpoint(
    endpoint_url=f"https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
    temperature = 0.1,
    max_new_tokens = 512,
)

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "table"],
    template="""
You are a BigQuery SQL assistant. The active table is `{table}`.

You have access to the following tools:
{tools}

Use this format:

Question: the question to answer
Thought: your reasoning process
Action: the action to take, should be one of [bigquery_query, describe_table]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
"""
)

# ------------------------
# 9. Initialize React Agent & AgentExecutor
# ------------------------

tools = [bigquery_tool, schema_tool]

prompt_with_vars = prompt.partial(
    table=active_table,
    tools="\n".join(f"{tool.name}: {tool.description}" for tool in tools),
    tool_names=[tool.name for tool in tools],
)

react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_with_vars,
)

agent_executor = AgentExecutor(agent=react_agent, tools=tools)

# ------------------------
# 10. Streamlit UI
# ------------------------
st.title("🩺 OMOP SynPUF Text-to-SQL (BigQuery)")
st.markdown(
    f"**Project**: `{PROJECT}`  &nbsp;|&nbsp;  "
    f"**Dataset**: `{DATASET}`  &nbsp;|&nbsp;  "
    f"**Table**: `{active_table}`"
)

user_input = st.text_area("Ask a question (natural language or SQL):")

if st.button("Run") and user_input.strip():
    with st.spinner("Thinking…"):
        try:
            # Invoke the agent with user input
            response = agent_executor.invoke({"input": user_input})
            st.success("Agent response")
            st.code(response["output"])
        except Exception as e:
            st.error(f"Agent error: {e}")

            
