
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

```bash
git clone https://github.com/your-username/omop-text-to-sql.git
cd omop-text-to-sql

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

2. Set up your environment variables or use a `.env` file:

```bash
export OPENAI_API_KEY=your_openai_api_key            # if using OpenAI
export HUGGINGFACEHUB_API_TOKEN=your_hf_token        # if using HuggingFace models
export GOOGLE_PROJECT_ID=your-google-project-id
export OMOP_DATASET_ID=synpuf
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

Alternatively, you can configure these using [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-cloud/secrets-management) when deploying.

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## Deployment Notes

- When deploying on [Streamlit Cloud](https://streamlit.io/cloud), add your secrets (API tokens, service account keys) in the app’s **Secrets** settings rather than committing them in code.
- Ensure the Google Cloud project and service account have proper BigQuery access.
- If you use Ollama as the LLM backend, you need to have Ollama installed and running on your machine or accessible endpoint.
- For Hugging Face models, verify your token has access to the specified model.

---

## Troubleshooting

- **Invalid JWT Signature** errors usually indicate an issue with your Google service account JSON file — check that the credentials are valid and the environment variable points to the correct file.
- **Agent errors related to LLM** often stem from incorrect model names or unsupported tasks. Verify the model and provider configurations.
- If you face API quota or billing issues, confirm your Google Cloud billing account is active.

---

## License

MIT License

---

If you find this project useful, please star ⭐️ the repo!

---

For questions or issues, please open an issue in the repo.
