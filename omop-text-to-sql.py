from __future__ import annotations
"""
Web​‑based Text​‑to​‑SQL agent for the CMS SynPUF OMOP dataset in BigQuery.
=======================================================================

This Streamlit app lets anyone ask natural​‑language questions that are
translated into SQL and executed against the SynPUF OMOP tables in
BigQuery.

Changes
-------
* **Credential handling**: Reads `GOOGLE_APPLICATION_CREDENTIALS_JSON` from
  Streamlit secrets, validates it, writes to `/tmp/gcp-key.json`, and sets
  `GOOGLE_APPLICATION_CREDENTIALS`.
* **Updated for LangChain 0.2.0​‑plus**: `SQLDatabase.from_uri` now expects
  the keyword `sample_rows_in_table_info`, not `sample_rows_in_table`.
* **Free hosted LLM**: Switched to Hugging Face Inference API (e.g.
  `meta-llama/Meta-Llama-3-8B-Instruct`) via `langchain_community.llms.HuggingFaceEndpoint`.
  Requires a free `HUGGINGFACEHUB_API_TOKEN`.
"""

import os
import json

import streamlit as st
#st.set_page_config(page_title="OMOP Text-to-SQL", layout="wide")

from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentType, create_sql_agent
from langchain_community.llms import HuggingFaceEndpoint

# -----------------------------------------------------------------------------
# Credential handling (BigQuery)
# -----------------------------------------------------------------------------
json_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
if json_creds:
    try:
        creds_dict = json.loads(json_creds)
    except json.JSONDecodeError:
        st.error(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON. "
            "Ensure the secret is a raw, multiline string containing the full key."
        )
        st.stop()

    creds_path = "/tmp/gcp-key.json"
    with open(creds_path, "w") as f:
        json.dump(creds_dict, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_sql_database(project_id: str, dataset_id: str, sample_rows: int = 3) -> SQLDatabase:
    """Return a LangChain SQLDatabase connected to BigQuery."""
    uri = f"bigquery://{project_id}/{dataset_id}"
    return SQLDatabase.from_uri(uri, sample_rows_in_table_info=sample_rows)


def build_agent(db: SQLDatabase, model_repo: str, temperature: float = 0.0):
    """Instantiate the HuggingFaceEndpoint LLM + LangChain SQL agent."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error(
            "HUGGINGFACEHUB_API_TOKEN is missing. Add it to Streamlit secrets. "
            "You can create a free token at https://huggingface.co/settings/tokens"
        )
        st.stop()

    llm = HuggingFaceEndpoint(
        repo_id=model_repo,
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        temperature=temperature,
        max_new_tokens=512
    )

    return create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    st.title("🔎 OMOP SynPUF Text-to-SQL Agent")
    st.markdown("Ask questions about the CMS SynPUF OMOP dataset.")

    # BigQuery env vars
    project_id = os.getenv("GOOGLE_PROJECT_ID", "fluid-catfish-456819-v2")
    dataset_id = os.getenv("OMOP_DATASET_ID", "synpuf")

    # LLM model (Hugging Face repo)
    model_repo = os.getenv("LLM_REPO", "meta-llama/Meta-Llama-3-8B-Instruct")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        st.error(
            "Google credentials not found. Add a service​‑account key via "
            "GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS."
        )
        st.stop()

    # Connect to BigQuery
    try:
        db = get_sql_database(project_id, dataset_id)
    except Exception as err:
        st.error(f"❌ Failed to connect to BigQuery: {err}")
        st.stop()

    # Build the agent
    agent = build_agent(db, model_repo, temperature=temperature)

    # UI
    user_query = st.text_input("❓ Enter your question:", placeholder="e.g. How many patients are over 65?")

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
