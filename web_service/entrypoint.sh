#!/bin/bash

echo "Waiting for database..."
while ! pg_isready -h $DB_HOST -p 5432 -U $DB_USER; do
  sleep 1
done

echo "Running migrations..."
python manage.py migrate --noinput

echo "Creating superuser if needed..."
python manage.py shell << END
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin')
    print('Superuser created: admin/admin')
else:
    print('Superuser already exists')
END

echo "Starting Django server..."
exec python manage.py runserver 0.0.0.0:8000