# Fahim AI: Smart Document Analyzer

**Fahim AI** is a professional Retrieval-Augmented Generation (RAG) application that allows users to chat with their documents (PDF, Word, PowerPoint, and Text) in both **Arabic** and **English**. It leverages the power of **Google Gemini 2.5 Flash Lite** and **Pinecone Vector Database** to provide accurate, context-aware answers.
---

## Collaboration & Open Source

* **Developed by**: This project is a collaborative effort by the **OSS Community** ' AI team.
* **Open Source**: Built with the spirit of open-source development, encouraging learning and community contribution.
* **Community Driven**: Created as part of the initiatives within the OSS community to build impactful AI tools.
---

## Key Features

* **Multilingual Support**: Fully optimized for Arabic (RTL support) and English.
* **Vector Search (RAG)**: Uses Pinecone for efficient semantic search to provide accurate context to the AI.
* **Fast Embeddings**: Powered by `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for high-quality multilingual vectorization.
* **Modern UI/UX**: A clean, responsive dashboard built with Streamlit, featuring a dedicated sidebar for document management.
* **Secure Config**: Implements best practices by using Streamlit Secrets to manage API keys.
---

## Tech Stack

* **Frontend**: Streamlit.
* **LLM**: Google Gemini 2.5 Flash Lite (via OpenRouter).
* **Vector DB**: Pinecone.
* **Orchestration**: LangChain.
* **Embedding Model**: Hugging Face Transformers.
---

## Project Structure

```text
FAHIM_AI/
├── .streamlit/        
     ├── secrets.toml  # Secret management APIs (excluded from Git)
├── database.py        # Vector DB and Embedding functions
├── utils.py           # Document processing and text splitting
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

## nstallation & Setup

### 1.Clone the repository:

```
git clone https://github.com/CS-Abdulaziz/fahim-ai-assistant.git
cd FAHIM_AI
```

### 2.Install dependencies:

```
pip install -r requirements.txt
```

### 3.Configure Secrets:

Create a folder `.streamlit` and a file `secrets.toml` inside it:

```
# .streamlit/secrets.toml
OPENROUTER_API_KEY = "your_key"
PINECONE_API_KEY = "your_key"
PINECONE_INDEX_NAME = "your_index"
```

### 4.Run the App:

```
streamlit run fahim_ai_app.py
```


