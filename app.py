import os
import re
import warnings
import pandas as pd
import backoff
from datetime import datetime
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from docling.document_converter import DocumentConverter
from opik import Opik, track, evaluate
from opik.evaluation.metrics import Hallucination, AnswerRelevance
from opik.evaluation import models
import litellm
import opik
from fastapi.responses import StreamingResponse
from litellm.integrations.opik.opik import OpikLogger
from litellm import completion, APIConnectionError
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Response

from langchain.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

def initialize_opik():
    opik_logger = OpikLogger()
    litellm.callbacks = [opik_logger]
    opik.configure(api_key=os.getenv("OPIK_API_KEY"),workspace=os.getenv("workspace"),force=True)


# Initialize Opik and load environment variables
load_dotenv()
initialize_opik()

# Initialize Opik Client
dataset = Opik().get_or_create_dataset(
    name="Cyfuture_faq",
    description="Dataset on IGL FAQ",
)

@app.post("/upload_dataset/")
def upload_dataset(file: UploadFile = File(...)):
    try:
        df = pd.read_excel(file.file)
        dataset.insert(df.to_dict(orient='records'))
        return {"message": "Dataset uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To use the uploaded dataset in the evaluation task manually
def upload_dataset():
    df = pd.read_excel("dataset.xlsx")
    dataset.insert(df.to_dict(orient='records'))
    return "Dataset uploaded successfully"

# Initialize LLM Models
model = ChatOllama(model="deepseek-r1:7b", base_url="http://localhost:11434", temperature=0.2, max_tokens=200)

def load_documents(folder_path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        all_documents = []
        os.makedirs('data', exist_ok=True)
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if filename.endswith('.pdf'):
                loader = PyMuPDFLoader(file_path)
            elif filename.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                continue  # Skip unsupported files
            
            documents = loader.load()
            all_documents.extend(text_splitter.split_documents(documents))
            print(f"Processed and indexed {filename}")
        
        return all_documents
# Vector Store Setup
def setup_vector_store(documents):
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("deepseek_cyfuture")
    return vectorstore


# Create RAG Chain
def create_rag_chain(retriever):
    prompt_template = ChatPromptTemplate.from_template(
        """
You are an AI questiona answering assistant specialized in answering user queries strictly from the provided context. Give detailed answer to user question considering the context.

STRICT RULES:
- You *must not* answer any questions outside the provided context.
- If the question is unrelated to billing, payments, customer, or meter reading, respond with exactly:
  **"This question is outside my specialized domain."**
- Do NOT attempt to generate an answer from loosely related context.
- If the context does not contain a valid answer, simply state: **"I don't know the answer."**

VALIDATION STEP:
1. Check if the query is related to **billing, payments, customer, or meter reading**.
2. If NOT, respond with: `"This question is outside my specialized domain."` and nothing else.
3. If the context does not contain relevant data try to find best possible answer from the context.
4. Do NOT generate speculative answers.
5. if the generated answer don't adress the question then try to find the best possible answer from the context you can add more releavnt context to the answer.

Question: {question}  
Context: {context}  
Answer:
        """
       
    )
    return (  
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt_template 
        | model 
        | StrOutputParser()
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def clean_response(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()



@track()
def llm_chain(input_text):
    try:
        context = "\n".join(doc.page_content for doc in retriever.invoke(input_text))
        response = "".join(chunk for chunk in rag_chain.stream(input_text) if isinstance(chunk, str))
        return {"response": clean_response(response), "context_used": context}
    except Exception as e:
        return {"error": str(e)}

def evaluation_task(x):
    try:
        result = llm_chain(x['user_question'])
        return {"input": x['user_question'], "output": result["response"], "context": result["context_used"], "expected": x['expected_output']}
    except Exception as e:
        return {"input": x['user_question'], "output": "", "context": x['expected_output']}

# experiment_name = f"Deepseek_{dataset.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
# metrics = [Hallucination(model=model1), AnswerRelevance(model=model1)]


@app.post("/run_evaluation/")
@backoff.on_exception(backoff.expo, (APIConnectionError, Exception), max_tries=3, max_time=300)
def run_evaluation():
    experiment_name = f"Deepseek_{dataset.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    metrics = [Hallucination(), AnswerRelevance()]
    try:
        evaluate(
            experiment_name=experiment_name,
            dataset=dataset,
            task=evaluation_task,
            scoring_metrics=metrics,
            experiment_config={"model": model},
            task_threads=2
        )
        return {"message": "Evaluation completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @backoff.on_exception(backoff.expo, (APIConnectionError, Exception), max_tries=3, max_time=300)
# def run_evaluation():
#     return evaluate(experiment_name=experiment_name, dataset=dataset, task=evaluation_task, scoring_metrics=metrics, experiment_config={"model": model}, task_threads=2)

# run_evaluation()

# Create Vector Database
def create_db():
    source = r'AI Agent'
    markdown_content = load_documents(source)
    setup_vector_store(markdown_content)
    return "Database created successfully"

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
vectorstore = FAISS.load_local("deepseek_cyfuture", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever( search_kwargs={'k': 2})
rag_chain = create_rag_chain(retriever)

@track()
@app.get("/query/")
def chain(input_text: str = Query(..., description="Enter your question")):
    try:
        # def generate():
        #     for chunk in rag_chain.stream(input_text):
        #         if isinstance(chunk, str):
        #             yield chunk
        def generate():
            buffer = ""  # Temporary buffer to hold chunks until `</think>` is found
            start_sending = False

            for chunk in rag_chain.stream(input_text):
                if isinstance(chunk, str):
                    buffer += chunk  # Append chunk to buffer
                    
                    # Check if `</think>` is found
                    if "</think>" in buffer:
                        start_sending = True
                        # Yield everything after `</think>` (including `</think>` itself)
                        yield buffer.split("</think>", 1)[1]
                        buffer = ""  # Clear the buffer after sending the first response
                    elif start_sending:
                        yield chunk  # Continue yielding after the `</think>` tag
        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Assistant API!"}

if __name__ == "__main__":
    # start my fastapi app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

    # questions=[ "Is the website accessible through mobile also? please tell the benefits of it","How do I register for a new connection?","how to make payments?",]
    # # Questions for retrieval
    # # Answer questions
    # create_db()
    # # Load Vector Store
    # embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    # vectorstore = FAISS.load_local("deepseek_cyfuture", embeddings, allow_dangerous_deserialization=True)
    # retriever = vectorstore.as_retriever( search_kwargs={'k': 3})
    # rag_chain = create_rag_chain(retriever)

    # for question in questions:
    #     print(f"Question: {question}")
    #     for chunk in rag_chain.stream(question):
    #         print(chunk, end="", flush=True)
    #     print("\n" + "-" * 50 + "\n")