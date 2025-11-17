FROM python:3.11

WORKDIR /app

# Copy the requirements file from ai_service folder
COPY ai_service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the ai_service code
COPY ai_service/. .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
