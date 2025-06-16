"""
Web-based Text‑to‑SQL agent for the CMS SynPUF OMOP dataset in BigQuery.
=======================================================================

This Streamlit app lets anyone ask natural‑language questions that are
translated into SQL and executed against the SynPUF OMOP tables in
BigQuery.

Run locally:
```bash
streamlit run omop_text_to_sql_agent.py
```
Deploy on Streamlit Cloud by adding your secrets (OpenAI key and the
service‑account JSON) – the code below handles writing the JSON to a
file at runtime so the BigQuery client can authenticate.
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
# Credential handling for Streamlit Cloud or any environment where the
# credentials are supplied as a JSON string in an environment variable.
# -----------------------------------------------------------------------------
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    creds_path = "/tmp/gcp-key.json"
    with open(creds_path, "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


def get_sql_database(
    project_id: str,
    dataset_id: str,
    sample_rows: int = 3,
) -> SQLDatabase:
    uri = f"bigquery://{project_id}/{dataset_id}"
    return SQLDatabase.from_uri(uri, sample_rows_in_table=sample_rows)


def build_agent(
    db: SQLDatabase,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
):
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    return create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )


def main():
    st.set_page_config(page_title="OMOP Text-to-SQL", layout="wide")
    st.title("🔎 OMOP SynPUF Text-to-SQL Agent")
    st.markdown("Ask questions about the CMS SynPUF OMOP dataset.")

    # Environment variables / secrets
    project_id = os.getenv("GOOGLE_PROJECT_ID", "fluid-catfish-456819-v2")
    dataset_id = os.getenv("OMOP_DATASET_ID", "synpuf")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Initialise database connection and agent
    db = get_sql_database(project_id, dataset_id)
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
