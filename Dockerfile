FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       curl \
       wget \
       git \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The default requirements install the lightweight app without TensorFlow.
COPY requirements.txt requirements-light.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY assets/ ./assets/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
