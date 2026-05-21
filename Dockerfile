FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /workspace/swarm_sim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libfreetype6 \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main_simulation.py", "--mode", "consensus", "--duration", "1", "--no-animate"]
