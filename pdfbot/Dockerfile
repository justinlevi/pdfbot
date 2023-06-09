# Use python:3.10 as base image for both stages
FROM python:3.10 as requirements-stage

# Set work directory
WORKDIR /tmp

# Install poetry
RUN pip install poetry

# Copy files
COPY ./pdfbot/pyproject.toml ./pdfbot/poetry.lock* /tmp/

# Export the dependencies in requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Now use the same python:3.10 image for final stage
FROM python:3.10

# Set work directory
WORKDIR /app

# Copy the requirements.txt file from the first stage
COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt

# Upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of your application
COPY ./pdfbot/ /app/

# Expose the required port
EXPOSE 8501

# Healthcheck command
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Default command, uses Streamlit
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
