from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_cohere import CohereEmbeddings

load_dotenv()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Initialize the Cohere model and embeddings
model = ChatCohere(model="command-a-03-2025", temperature=0.7)
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Step 3: Create a Chroma vector store and add documents
vector_store = Chroma(
    collection_name="langchain_docs",
    embedding_function=embeddings
)

# Step 4: Add documents to the vector store
vector_store.add_documents(documents)

# Step 04: Convert vector store documents to a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

question = "What does LangChain do?"
results = retriever.invoke(question)

print("🤖🤖🤖retrieved documents:", results)

for i, doc in enumerate(results):
    print(f"📖📖📖Document {i+1}: {doc.page_content}")

'''
# Step 05: Retrieve relevant documents based on the question
retrieved_docs = retriever.get_relevant_documents(question)
'''