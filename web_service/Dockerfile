# ./web_service/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire Django project
COPY . .

# Create necessary directories
RUN mkdir -p /shared/images /shared/audio staticfiles

# Collect static files (will run even if it fails)
RUN python manage.py collectstatic --noinput || true

EXPOSE 8000

# Start script that runs migrations then starts server
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]