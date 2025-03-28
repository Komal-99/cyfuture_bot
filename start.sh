#!/bin/bash

# Set environment variables for optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1

# Start Ollama in the background
ollama serve &

# Pull the model if not already present
if ! ollama list | grep -q "deepseek-r1:7b"; then
    ollama pull deepseek-r1:7b
fi
if ! ollama list | grep -q "nomic-embed-text"; then
    ollama pull nomic-embed-text
fi
# Wait for Ollama to start up
max_attempts=30
attempt=0
while ! curl -s http://localhost:11434/api/tags >/dev/null; do
    sleep 1
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "Ollama failed to start within 30 seconds. Exiting."
        exit 1
    fi
done

echo "Ollama is ready."

# Print the API URL
echo "API is running on: http://0.0.0.0:7860"

# Start FastAPI in the background
uvicorn app:app --host 0.0.0.0 --port 7860 --workers 4 --limit-concurrency 20 &

# Start Streamlit for UI
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0