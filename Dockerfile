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

# Remove any GUI OpenCV if a dependency tries to bring it in
RUN pip uninstall -y opencv-python opencv-contrib-python || true

RUN pip install --no-cache-dir -r requirements.txt

# Double-check final installed OpenCV packages
RUN pip show opencv-python opencv-python-headless opencv-contrib-python || true

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
