FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app /saisresult && \
    apt-get update && \
    apt-get install -y tini bash python3 python3-pip libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT [" /sbin/tini\, \--\, \bash\, \/app/run.sh\]
