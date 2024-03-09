# Base image with desired Python version
FROM python:3.12-slim

# Create a directory for the application
WORKDIR /app

# Copy pyproject.toml to configure Poetry
COPY pyproject.toml ./

# Install Poetry (if not already installed in the base image)
RUN pip install poetry

# Configure Poetry not to create a virtual environment (we're using Docker)
RUN poetry config virtualenvs.create false

# Install dependencies (excluding development ones)
RUN poetry install --no-dev

# Copy your application code
COPY . .

# Define the command to run your application (replace with your actual command)
CMD ["python", "main.py"]