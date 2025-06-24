# Base image
FROM python:3.10-bullseye


# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    binutils \
    && rm -rf /var/lib/apt/lists/* --verbose


# Set working directory
WORKDIR /app

# Copy files
COPY ./app ./app
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt 

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
