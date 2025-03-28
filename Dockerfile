FROM python:3.11.4-slim-buster



# Install curl and Ollama
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.ai/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up user and environment
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .


COPY --chown=user . .

# Make the start script executable
RUN chmod +x start.sh
# Expose FastAPI & Streamlit ports
EXPOSE 7860 8501

CMD ["./start.sh"]