"""
Web-based Text‑to‑SQL agent for the CMS SynPUF OMOP dataset in BigQuery.
=======================================================================

This Streamlit app lets anyone ask natural‑language questions that are
translated into SQL and executed against the SynPUF OMOP tables in
BigQuery.

**Credential logic updated** ➜ if `GOOGLE_APPLICATION_CREDENTIALS_JSON`
string is provided (e.g. via Streamlit secrets), the app now **validates**
it, writes it to `/tmp/gcp-key.json`, and sets the
`GOOGLE_APPLICATION_CREDENTIALS` env var before BigQuery is initialised.
"""
from __future__ import annotations

import os
import json
from typing import Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, create_sql_agent
from langchain_community.utilities import SQLDatabase

# -----------------------------------------------------------------------------
# Credential handling for Streamlit Cloud or any environment where credentials
# are supplied as a JSON string in an environment variable.
# -----------------------------------------------------------------------------
json_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
if json_creds:
    try:
        # Validate JSON early so we fail fast if formatting is wrong
        creds_dict = json.loads(json_creds)
    except json.JSONDecodeError:
        st.error(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON. "
            "Double‑check your Streamlit secret – it should be a raw, multiline string "
            "containing the full service‑account key."
        )
        st.stop()

    creds_path = "/tmp/gcp-key.json"
    with open(creds_path, "w") as f:
        json.dump(creds_dict, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_sql_database(
    project_id: str,
    dataset_id: str,
    sample_rows: int = 3,
) -> SQLDatabase:
    """Return a LangChain SQLDatabase bound to BigQuery."""
    uri = f"bigquery://{project_id}/{dataset_id}"
    return SQLDatabase.from_uri(uri, sample_rows_in_table=sample_rows)


def build_agent(
    db: SQLDatabase,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
):
    """Instantiate a LangChain SQL agent."""
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    return create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="OMOP Text-to-SQL", layout="wide")
    st.title("🔎 OMOP SynPUF Text-to-SQL Agent")
    st.markdown("Ask questions about the CMS SynPUF OMOP dataset.")

    # Environment variables / secrets
    project_id = os.getenv("GOOGLE_PROJECT_ID", "fluid-catfish-456819-v2")
    dataset_id = os.getenv("OMOP_DATASET_ID", "synpuf")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Check that ADC path is set before touching BigQuery
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        st.error(
            "Google credentials not found. Set either GOOGLE_APPLICATION_CREDENTIALS_JSON "
            "(preferred on Streamlit Cloud) or GOOGLE_APPLICATION_CREDENTIALS (path to a key file)."
        )
        st.stop()

    # Initialise database connection and agent
    try:
        db = get_sql_database(project_id, dataset_id)
    except Exception as err:
        st.error(f"❌ Failed to connect to BigQuery: {err}")
        st.stop()

    agent = build_agent(db, model_name=model_name)

    # --- UI ---
    user_query = st.text_input(
        "❓ Enter your question:",
        placeholder="e.g. How many patients are over 65?",
    )

    if st.button("Submit", use_container_width=True) and user_query:
        with st.spinner("Running query..."):
            try:
                answer = agent.run(user_query)
                st.success("✅ Response")
                st.write(answer)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")


if __name__ == "__main__":
    main()