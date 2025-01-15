# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the app code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for uvicorn
EXPOSE 8001

# Command to run the AI service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
