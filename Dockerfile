# Use a lightweight Python base image
FROM python:3.13-slim

# Copy the requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy your operator code into the container
COPY operator.py .

# Use the kopf run command to execute the operator
CMD ["kopf", "run", "--namespace=default", "--verbose", "operator.py"]