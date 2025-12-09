FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONPATH=/app \
	PORT=8501

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends build-essential g++ \
	&& pip install --no-cache-dir -r requirements.txt \
	&& apt-get purge -y --auto-remove build-essential g++ \
	&& rm -rf /var/lib/apt/lists/*

COPY common ./common
COPY elector ./elector
COPY artifacts ./artifacts

WORKDIR /app/elector

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]