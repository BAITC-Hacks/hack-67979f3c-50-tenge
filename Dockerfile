FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY starter/requirements.txt starter/requirements.txt
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY starter/ starter/
COPY run.py .
COPY pipeline/ pipeline/
COPY viz/ viz/
COPY .streamlit/config.toml .streamlit/config.toml

CMD ["python", "run.py", "--data", "/app/data", "--out", "/app/out"]
