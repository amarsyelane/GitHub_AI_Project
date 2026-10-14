
from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
import json
from time import sleep

load_dotenv()

question = "Who is the prime minister of India?"

def retrieve_wikipedia_docs(question, max_retries=3):
    """Retrieve documents from Wikipedia with retry logic and better error handling."""
    retriever = WikipediaRetriever(top_k_results=2, lang="en")
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            docs = retriever.invoke(question)
            
            if not docs:
                print("Warning: No documents found for the query.")
                return None
            
            return docs
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}. This may be a Wikipedia API issue.")
            if attempt < max_retries - 1:
                sleep(2)  # Wait before retrying
                continue
            else:
                raise
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                sleep(2)
                continue
            else:
                raise

try:
    docs = retrieve_wikipedia_docs(question)
    
    if docs:
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i + 1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:500]}...")  # Print first 500 chars
    
except Exception as e:
    print(f"Failed to retrieve documents after retries: {e}")