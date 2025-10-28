# Use a lightweight Python base image
FROM python:3.13-slim

# Install kubectl
RUN apt-get update && apt-get install -y curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl

# Install helm since the operator may need it to manage charts inside the controller
RUN apt-get update && apt-get install -y curl tar ca-certificates && \
    curl -L https://get.helm.sh/helm-v3.14.4-linux-amd64.tar.gz -o /tmp/helm.tgz && \
    tar -xzf /tmp/helm.tgz -C /tmp && mv /tmp/linux-amd64/helm /usr/local/bin/helm && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Installing kopf and other dependencies for the operator to function
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy your operator code into the container
COPY operator.py .

# Use the kopf run command to execute the operator
CMD ["kopf", "run", "--namespace=default", "--verbose", "operator.py"]