# Use a lightweight Python base image
FROM python:3.13-slim

# Install the Kopf framework and any other dependencies
RUN pip install kopf

# Set the working directory
WORKDIR /app

# Copy your operator code into the container
COPY operator.py .

# Use the kopf run command to execute the operator
CMD ["kopf", "run", "--namespace=default", "--verbose", "operator.py"]