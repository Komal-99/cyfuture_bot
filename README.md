# AI Assistant API

## 🚀 Overview

This project is an AI-powered assistant that uses FastAPI and FAISS for retrieval-augmented generation (RAG). It processes user queries using a vector database and evaluates responses with Opik.

## 🛠️ Features

- Upload and manage datasets
- Query AI assistant with domain-specific constraints
- Use FAISS for efficient document retrieval
- Evaluate LLM responses using Opik

## 📽️ Demo Video

[🎥 Click here to watch the demo](https://drive.google.com/file/d/10h4VnTm_y5SBczI6NnoTuqRxyq55HAn5/view?usp=sharing)


## 📦 Installation

### Install Ollama

Ollama is required for this project. Follow these steps to install it:

```bash
# For macOS
brew install ollama

# For Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version

# Windows
You can download from web https://ollama.com/
```
# Clone and Setup the Project
## Clone the repository
```
git clone https://github.com/Komal-99/cyfuture_bot.git
```

## Navigate to the project directory
```
cd cyfuture_bot
```

## Install dependencies
```
pip install -r requirements.txt  # For Python projects
yarn install  # For JavaScript projects
 ```

🚀 Usage
Start the Project
Run the ```start.sh``` script to set up and launch the application:
```
chmod +x start.sh
./start.sh

```
This script:

Sets environment variables for optimization

Starts Ollama in the background

Pulls required models (deepseek-r1:7b, nomic-embed-text)

Waits for Ollama to initialize

### Launches the FastAPI server on http://127.0.0.1:7860
###  Streamlit Application - http://127.0.0.1:8501

## API Endpoints
Upload Dataset
```
POST /upload_dataset/ #Upload an Excel dataset to be used for evaluation.
```
Run Evaluation
```
POST /run_evaluation/ #Evaluate the model's performance using Opik.

```
Query AI Assistant
```
GET /query/?input_text=your_question # Ask the assistant a question. The model retrieves relevant information and generates an answer based on indexed documents.

```
📂 Folder Structure

```
.
├── AI_Agent/          # Datasource
├── deepseek_cyfuture/ # DeepSeek Vector db
├── .env               # Environment variables
├── .gitignore         # Files to ignore in Git
├── dataset.xlsx       # Sample dataset file
├── Dockerfile         # Docker configuration
├── requirements.txt   # Dependencies (Python projects)
├── start.sh           # Startup script
├── app.py             # Main application file
├── README.md          # Project documentation

```
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a new branch (git checkout -b feature-branch)

Commit your changes (git commit -m 'Add new feature')

Push to the branch (git push origin feature-branch)

Create a pull request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

📬 Contact
For questions or issues, reach out:

GitHub: https://github.com/Komal-99

