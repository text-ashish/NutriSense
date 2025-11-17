FROM python:3.11

WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Expose port
EXPOSE 8001

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
