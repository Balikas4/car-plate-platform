# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the app code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Django
EXPOSE 8000

# Command to run Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
