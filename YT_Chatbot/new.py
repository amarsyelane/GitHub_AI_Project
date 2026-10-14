from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM
llm = ChatCohere(model="command-a-03-2025", temperature=0.7)

# Step 01 A: Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo"
transcript = ""  # FIX 1: Initialize transcript variable

try:
    # Fetch the transcript for the given YouTube video ID
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)

    # Combine the transcript chunks into a single string    
    transcript = " ".join(
        [chunk.text for chunk in transcript_list]  # Access .text attribute
    )
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
    print("Please check the video settings or try a different video.")
    exit(1)
except (NoTranscriptFound, VideoUnavailable) as e:
    print(f"Error fetching transcript: {e}")
    exit(1)

# Step 01 B: Indexing (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.create_documents([transcript])

# Step 01 C & D: Indexing (embedding generation and storing in vector store)
# embeddings
embedding = CohereEmbeddings(model="embed-english-v3.0")

# Create a vector store
vector_store = FAISS.from_documents(chunks, embedding)

# Step 02 Retrieval (Querying)
# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Step 03 Automation (Prompting)
# Create a prompt template for the question-answering task
prompt = PromptTemplate(
    template="""You are a helpful assistant. Answer only based on the transcript context provided. If the information is not in the context, say you don't know.
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=['context', 'question'],
)

def format_retrieved_docs(retrieved_docs):
    """Format retrieved documents into a single context string."""
    if not retrieved_docs:
        return "No relevant context found."
    # Join all documents with clear separation
    context_text = "\n\n".join([chunk.page_content for chunk in retrieved_docs])
    return context_text

# Build the chain: retriever -> format -> add to dict
retrieval_chain = retriever | RunnableLambda(format_retrieved_docs)

# Create parallel chain that formats input properly
def prepare_input(question):
    """Prepare the input for the prompt by retrieving context."""
    context = retrieval_chain.invoke(question)
    return {"context": context, "question": question}

parser = StrOutputParser()

#Use simpler chain that properly handles string formatting
main_chain = RunnableLambda(prepare_input) | prompt | llm | parser

# Invoke the chain
result = main_chain.invoke("Advice for young people")
print("\n" + "="*70)
print("ANSWER:")
print("="*70)
print(result)