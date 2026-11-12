FROM python:3.12-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalação das dependências Python
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY main.py .

# Usuário sem privilégios
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
