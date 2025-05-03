# Brok-rag-v4.5
Brok v4.5 — RAG Pipeline Supercharged with Mistral-7B + LangChain + FAISS

> 🚀 Real-time Knowledge Retrieval + Generative Reasoning. The power of search meets the depth of language models.

---

## 🧠 What is Brok v4.5?

**Brok v4.5** is a fully integrated **Retrieval-Augmented Generation (RAG)** pipeline combining:

- **Live Web Search** using [Tavily](https://tavily.com/)
- **High-Quality Embeddings** via `all-MiniLM-L6-v2`
- **Lightning-fast Vector Retrieval** with `FAISS`
- **Multi-source Web Scraping** via `LangChain WebBaseLoader`
- **Generation Engine** powered by **Mistral 7B Instruct**
- **LangChain's RetrievalQA** for seamless orchestration

> Brok v4.5 is not just a chatbot. It’s your custom research agent, financial analyst, news summarizer, meme hunter, and policy explainer—*in one pipeline.*

---

## 🎯 Key Features

- 🌐 **Live Web Search**: Pulls from 20+ trusted sources (BBC, Bloomberg, RBI, etc.)
- 🔍 **Document Chunking + Vectorization**: Efficiently splits, embeds, and indexes data for semantic search.
- 🧬 **LLM-Powered Answering**: Uses Mistral-7B for natural, conversational, and detailed answers.
- 🗂 **RAG Architecture**: Augments the LLM with real data retrieved at runtime.
- 💬 **Source Transparency**: Returns cited sources used to generate every response.
- ⚙️ **Plug-and-Play Modularity**: Easy to extend, retrain, or scale.

---

## 🧰 Tech Stack

| Layer            | Tool/Library                      |
|------------------|-----------------------------------|
| LLM              | `Mistral-7B-Instruct` (HuggingFace) |
| Embedding Model  | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector Store     | `FAISS`                           |
| Search Tool      | `Tavily API`                      |
| Chunking         | `RecursiveCharacterTextSplitter` |
| Pipeline Orchestration | `LangChain`                |
| Data Sources     | 20+ Web URLs (news, finance, memes, tech) |
| Environment      | Python 3.10+, PyTorch, Transformers |

---

## ⚡ Quickstart

### 1. 🔧 Clone the Repo

```bash
git clone https://github.com/yourusername/brok-rag-v4.5.git
cd brok-rag-v4.5
