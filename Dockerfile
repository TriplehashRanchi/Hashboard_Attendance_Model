FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    cmake \
    pkg-config \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython

RUN pip install --no-cache-dir -r requirements.txt

# Remove GUI OpenCV, reinstall headless
RUN pip uninstall -y opencv-python opencv-contrib-python || true
RUN pip install --no-cache-dir opencv-python-headless==4.13.0.90

# Pre-download the insightface model during build
RUN mkdir -p /app/models && \
    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='/app/models').prepare(ctx_id=-1)"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
