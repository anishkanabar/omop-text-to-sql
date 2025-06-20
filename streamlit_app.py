# streamlit_app.py
import os
import json
import re
import streamlit as st
from google.cloud import bigquery
from langchain.llms import Ollama
from langchain.agents import Tool, AgentType, initialize_agent

# --------------------------------------------------
# 0. Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="OMOP SynPUF Text‑to‑SQL on BigQuery",
    layout="wide",
)

# --------------------------------------------------
# 1. BigQuery credentials & constants
# --------------------------------------------------
# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config["GOOGLE_APPLICATION_CREDENTIALS"]

# Assign variables
PROJECT = config["PROJECT"]
DATASET = config["DATASET"]
BQ_PATH  = f"{PROJECT}.{DATASET}"

client = bigquery.Client()

# --------------------------------------------------
# 2. Cached helper – list tables
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def list_tables():
    return sorted(tbl.table_id for tbl in client.list_tables(BQ_PATH))

tables = list_tables()

# --------------------------------------------------
# 3. Helper – run BigQuery queries
# --------------------------------------------------
def run_bigquery(sql: str):
    job = client.query(sql, location="US")
    return [dict(r) for r in job.result()]

# --------------------------------------------------
# 4. Sanitize utility
# --------------------------------------------------
def _sanitize(name: str) -> str:
    """Strip quotes, back‑ticks, whitespace/newlines."""
    return re.sub(r"[`\s\'\"]+", "", name)

# --------------------------------------------------
# 5. Tool A – SQL query tool
# --------------------------------------------------
def sql_tool(sql: str) -> str:
    sql = sql.replace("\n", " ")
    try:
        rows = run_bigquery(sql)
        return "Query returned no rows." if not rows else str(rows[:5])
    except Exception as e:
        return f"Query error: {e}"

bigquery_tool = Tool(
    name="bigquery_query",
    func=sql_tool,
    description=(
        f"Run StandardSQL against tables in `{BQ_PATH}` and return the first 5 rows. "
        "Query must use fully qualified table names like `project.dataset.table`."
    ),
)

# --------------------------------------------------
# 6. Tool B – List tables tool
# --------------------------------------------------
def list_tables_tool(_: str = "") -> str:
    return "\n".join(tables)

# Correct tool definition
list_tables_tool = Tool(
    name="list_tables",
    func=list_tables_tool,
    description="List all available tables in the dataset."
)

# --------------------------------------------------
# 7. Tool C – describe table schema
# --------------------------------------------------
def describe_table(table: str) -> str:
    table = _sanitize(table)
    if "." in table:
        return "❌ Pass only the table name, e.g. 'person'."
    try:
        tbl = client.get_table(f"{BQ_PATH}.{table}")
        lines = [f"{f.name} ({f.field_type})" for f in tbl.schema]
        return f"Schema for `{table}`:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error describing table `{table}`: {e}"

schema_tool = Tool(
    name="describe_table",
    func=describe_table,
    description="Return columns & types for a table (e.g. 'person').",
)


# --------------------------------------------------
# 8. Tool D – concept lookup
# --------------------------------------------------
def lookup_concept(term: str) -> str:
    """Search OMOP concept table for standard concepts by name."""
    sql = f"""
        SELECT concept_id, concept_name, vocabulary_id, domain_id
        FROM `{BQ_PATH}.concept`
        WHERE LOWER(concept_name) LIKE '%{term.lower()}%'
        AND standard_concept = 'S'
        ORDER BY LENGTH(concept_name)
        LIMIT 5
    """
    try:
        rows = run_bigquery(sql)
        if not rows:
            return f"No matching standard concepts found for '{term}'."
        return "\n".join(
            f"{r['concept_id']}: {r['concept_name']} ({r['vocabulary_id']}, {r['domain_id']})"
            for r in rows
        )
    except Exception as e:
        return f"Concept lookup error: {e}"

concept_lookup_tool = Tool(
    name="lookup_concept",
    func=lookup_concept,
    description=(
        "Use this to search the OMOP concept table for standard concepts by keyword. "
        "Input should be a term like 'ecg', 'blood pressure', 'glucose'. "
        "Returns concept_id and details."
    ),
)

# --------------------------------------------------
# 9. LLM & agent
# --------------------------------------------------
llm = Ollama(model="llama3")

system_msg = (
    "You are a BigQuery SQL assistant for the OMOP SynPUF dataset. "
    f"The project is `{PROJECT}` and the dataset is `{DATASET}`. "
    "You must only reference tables in this dataset. "
    "Use fully qualified table names like `project.dataset.table`. "
    "If unsure of the correct table name, call `list_tables()`. "
    "You may join across tables as needed. "
    "If you need to check column names, call `describe_table('table')`."
    "You can use `lookup_concept('keyword')` to find standard concept IDs from the OMOP concept table. "
    "Use the returned concept_id in queries, for example in `procedure_occurrence.procedure_concept_id`."
)

agent = initialize_agent(
    tools=[bigquery_tool, schema_tool, concept_lookup_tool],
    llm=llm,
    agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    system_message=system_msg,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
)

# --------------------------------------------------
# 10. Streamlit UI
# --------------------------------------------------
st.title("🩺 OMOP SynPUF Text‑to‑SQL (BigQuery)")
st.markdown(
    f"**Project**: `{PROJECT}`  &nbsp;|&nbsp;  "
    f"**Dataset**: `{DATASET}`  &nbsp;|&nbsp;  "
)

user_input = st.text_area("Ask a question (natural language or SQL):")

if st.button("Run") and user_input.strip():
    with st.spinner("Thinking…"):
        try:
            response_dict = agent.invoke({"input": user_input})
            full_answer = response_dict.get("output", "No response.")

            # Extract only the final answer, stripping all reasoning steps
            final_match = re.search(r"Final Answer:\s*(.*)", full_answer, re.DOTALL | re.IGNORECASE)
            answer = final_match.group(1).strip() if final_match else full_answer.strip()

            st.success("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Agent error: {e}")