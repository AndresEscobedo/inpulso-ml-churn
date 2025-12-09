FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONPATH=/app \
	PORT=5001

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends build-essential g++ \
	&& pip install --no-cache-dir -r requirements.txt \
	&& apt-get purge -y --auto-remove build-essential g++ \
	&& rm -rf /var/lib/apt/lists/*

COPY common ./common
COPY artifacts ./artifacts
COPY canary_model ./canary_model

WORKDIR /app/canary_model

EXPOSE 5001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]