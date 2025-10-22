FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# (optional but helpful for psycopg etc.)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# app code (copies everything, including api/app/)
COPY . .

EXPOSE 8000
# if your FastAPI app is at api/app/main.py with "app = FastAPI()":
CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","-w","2","-b","0.0.0.0:8000","api.app.main:app"]