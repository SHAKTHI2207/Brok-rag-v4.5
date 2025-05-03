from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os



# -------------------- SETUP ENVIRON --------------------
os.environ["TAVILY_API_KEY"] = "enter api key here"
os.environ["USER_AGENT"] = "brok-ai-dev/1.0"

# -------------------- 1. Web Search Tool --------------------
search = TavilySearchResults(k=5)

# -------------------- 2. Embedder --------------------
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------- 3. Load & Chunk Web Data --------------------

urls = [
    # List of URLs
    "https://www.xe.com/currencyconverter/convert/?Amount=1&From=GBP&To=INR",  # Currency exchange rate
    "https://www.reuters.com",  # Reuters news (financials, world news)
    "https://www.bbc.com/news",  # BBC News (general news and trends)
    "https://www.bloomberg.com",  # Bloomberg (business and financial news)
    "https://www.rbi.org.in",  # Reserve Bank of India (official financial data)
    "https://www.fca.org.uk",  # Financial Conduct Authority (UK regulations)
    "https://www.sec.gov",  # U.S. Securities and Exchange Commission (finance regulations)
    "https://www.mckinsey.com",  # McKinsey & Co. (business strategy and marketing)
    "https://www.forbes.com",  # Forbes (business, marketing, tech trends)
    "https://www.gartner.com",  # Gartner (IT and marketing insights)
    "https://www.techcrunch.com",  # TechCrunch (startup and tech news)
    "https://www.theverge.com",  # The Verge (tech and culture news)
    "https://www.cnbc.com",  # CNBC (market updates and financial news)
    "https://www.wsj.com",  # Wall Street Journal (business news)
    "https://www.ft.com",  # Financial Times (market analysis and news)
    "https://www.wikipedia.org",  # Wikipedia (general knowledge)
    "https://www.prsindia.org",  # PRS India (policy research)
    "https://x.com/memes" ,       ## Memes
    "https://9gag.com/trending", ## Memes
    "https://news.google.com" ,  ## GOOGLE NEWS

]

# Use the WebBaseLoader to load data from the URLs
loader = WebBaseLoader(urls)
docs = loader.load()

# -------------------- 4. Split Documents into Chunks --------------------

# Initialize the text splitter with a chunk size and overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split the loaded documents into smaller chunks
chunks = splitter.split_documents(docs)

# Optional: Preview the first few chunks to check the splitting result
print(chunks[:3])

# Use the WebBaseLoader to load data from the URLs
loader = WebBaseLoader(urls)

# Load the data and prepare documents for processing
docs = loader.load()

# Optional: Preview the first few docs to check the structure
print(docs[:3])



# -------------------- 4. Create Vectorstore --------------------
vectorstore = FAISS.from_documents(chunks, embedder)

# -------------------- 5. Load Mistral 7B using hugging face  --------------------

from huggingface_hub import login
login("enter api key here")
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

mistral_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=mistral_pipeline)

# -------------------- 6. Build RAG Chain --------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# -------------------- 7. Ask a Question --------------------
query = "What is the current pound rate in inr?"
result = qa_chain(query)

print("\n💬 Answer:", result["result"])
print("\n🔗 Sources:", [doc.metadata["source"] for doc in result["source_documents"]])
