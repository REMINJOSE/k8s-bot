FROM python:3.11-slim

# Install system dependencies, curl, and kubectl
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    ARCH=$(dpkg --print-architecture) && \
    # Install k8sgpt
    curl -LO "https://github.com/k8sgpt-ai/k8sgpt/releases/latest/download/k8sgpt_${ARCH}.deb" && \
    dpkg -i "k8sgpt_${ARCH}.deb" && \
    rm "k8sgpt_${ARCH}.deb" && \
    # Install official kubectl
    KUBECTL_VERSION=$(curl -L -s https://dl.k8s.io/release/stable.txt) && \
    curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${ARCH}/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
RUN pip install --no-cache-dir discord.py requests

# Copy bot code
COPY bot.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]
