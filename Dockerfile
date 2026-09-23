FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY starter/requirements.txt starter/requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY starter/ starter/

CMD ["python", "starter/starter.py", "--data", "/app/data", "--out", "/app/out"]
