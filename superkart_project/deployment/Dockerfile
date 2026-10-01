FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies first to improve Docker build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the non-root user expected by Hugging Face Spaces
RUN useradd -m -u 1000 user

# Copy application files
COPY --chown=user:user . /home/user/app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
