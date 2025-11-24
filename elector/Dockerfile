FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONPATH=/app \
	PORT=5002

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY elector ./elector

WORKDIR /app/elector

EXPOSE 5002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002"]