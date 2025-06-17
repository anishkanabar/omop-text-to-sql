# 👉 run this once before building the agent
from sqlalchemy import create_engine, inspect
import os
import streamlit as st

db_uri = st.text_input("DB URI", value=os.getenv("OMOP_DB_URI", "sqlite:///omop.db"))
engine = create_engine(db_uri)
print("Tables:", inspect(engine).get_table_names())
