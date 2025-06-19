# OMOP SynPUF Text-to-SQL Agent 🌐

A Streamlit web app that lets you ask natural language questions and runs LLM-generated SQL against the CMS SynPUF OMOP dataset in BigQuery.

![screenshot](examples/ex1.png)

---

## Requirements

- A Google Cloud project with BigQuery enabled and access to the OMOP SynPUF dataset.
- A Google service account JSON key with BigQuery permissions.
- [Ollama](https://ollama.com) installed and running locally (optional, if you use Ollama as your LLM backend).
- Hugging Face API token (if using HuggingFaceHub or HuggingFaceEndpoint as your LLM).

---

## Quick Start (Local)

1. Clone the repository and create a Python virtual environment:

git clone https://github.com/your-username/omop-text-to-sql.git
cd omop-text-to-sql

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

2. Set up your environment variables or use a .env file:
export OPENAI_API_KEY=your_openai_api_key            # if using OpenAI
export HUGGINGFACEHUB_API_TOKEN=your_hf_token        # if using HuggingFace models
export GOOGLE_PROJECT_ID=your-google-project-id
export OMOP_DATASET_ID=synpuf
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
Alternatively, you can configure these using Streamlit Secrets Management when deploying.

3. Run the Streamlit app:
streamlit run streamlit_app.py
