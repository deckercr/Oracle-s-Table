FROM python:3.10-slim

WORKDIR /app

# Install system tools needed for Python DB and Docker connection
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We don't copy the code here because we use a Volume in docker-compose
# This allows you to edit code on your computer and see changes instantly.

# Run Django on port 8000 (internal)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]