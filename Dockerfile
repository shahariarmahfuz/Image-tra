FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Download ResNet model
RUN wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_weights_tf_dim_ordering_tf_kernels.h5

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create upload directory
RUN mkdir -p static/uploads

EXPOSE 5000

CMD ["python", "app.py"]
