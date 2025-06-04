# Usa Python 3.11 slim
FROM python:3.11-slim

# Evita .pyc y hace flush inmediato de stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Copia e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Instala ffmpeg (para pydub)
RUN apt-get update && apt-get install -y ffmpeg

# Copia todo tu proyecto (incluye la carpeta static y tu app)
COPY . .

# (Opcional) Verifica que /app/static existe y contiene tus .mp3
RUN ls -l /app/static

# Expone el puerto 8080 (requerido por Cloud Run)
EXPOSE 8080

# Inicia la aplicación con uvicorn
CMD uvicorn jesy_ai.app:app --host 0.0.0.0 --port ${PORT:-8080}