# Use slim Python image
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install deps first (faster layer caching)
RUN pip install --no-cache-dir -r requirements.txt

# Default Streamlit port
ENV PORT=8501
EXPOSE 8501

CMD ["streamlit", "run", "omop_text_to_sql_agent.py", "--server.port=8501", "--server.enableCORS=false"]
