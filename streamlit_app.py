import streamlit as st
import os

# Must be the first Streamlit command
st.set_page_config(page_title="OMOP Text-to-SQL", layout="wide")

from langchain.llms import Ollama
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

# Page title and description
st.title("OMOP Text-to-SQL Agent")
st.markdown("Ask natural language questions and convert them to SQL over the OMOP database.")

# Sidebar for configuration
with st.sidebar:
    st.header("LLM & Database Config")
    model = st.text_input("LLM Model (e.g., llama3)", value="llama3")
    db_uri = st.text_input("DB URI", value=os.getenv("OMOP_DB_URI", "sqlite:///omop.db"))

# Button to connect and generate the agent
if st.button("Connect and Start Agent"):
    with st.spinner("Setting up agent..."):
        try:
            llm = Ollama(model=model)
            db = SQLDatabase.from_uri(db_uri)
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
            st.success("Agent ready! Ask your question below.")
            st.session_state.agent = agent_executor
        except Exception as e:
            st.error(f"Failed to initialize: {e}")

# Text input for query
if "agent" in st.session_state:
    user_query = st.text_input("Enter your question")
    if user_query:
        with st.spinner("Generating SQL..."):
            try:
                result = st.session_state.agent.run(user_query)
                st.code(result, language="sql")
            except Exception as e:
                st.error(f"Error running query: {e}")
else:
    st.info("Start the agent to enable question input.")
