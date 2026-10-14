from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Initialize the Cohere model and embeddings
load_dotenv()
model = ChatCohere(model="command-a-03-2025", temperature=0.7)
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Step 2: Create the FAISS vector store from documents
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

# enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3, 
        #"fetch k": 5, 
        "lambda_mult": 0.5
        }
    )
# Step 3: Retrieve relevant documents based on the question
question = "What does LangChain do?"
results = retriever.invoke(question)

print("🤖🤖🤖retrieved documents:", results)

for i, doc in enumerate(results):
    print(f"📖📖📖Document {i+1}: {doc.page_content}")