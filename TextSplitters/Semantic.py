# RecursiveCharacterTextSplitter    
import os
from unittest import result

from langchain_text_splitters import Language, Language, RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredFileLoader, UnstructuredURLLoader, PyPDFLoader, OnlinePDFLoader, CSVLoader, JSONLoader, PythonLoader
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

load_dotenv()

# embeddings
embedding = CohereEmbeddings(model="embed-english-v3.0")

# Initialize the SemanticChunker
text_splitter = SemanticChunker(
    embeddings=embedding, 
    breakpoint_threshold_type="standard_deviation",  # Options: "percentile", "standard_deviation", or "interquartile"
    breakpoint_threshold_amount=1.0  # This value will depend on the breakpoint_threshold_type you choose
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for
the next season. The sun was bright, and the air smelled of earth and fresh grass.
The Indian Premier League (IPL) is the biggest cricket league in the world. People
all over the world watch the matches and cheer for their favorite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates
fear in cities and villages. When such attacks happen, they leave behind pain and
sadness. To fight terrorism, we need strong laws, alert security forces, and support
from people who care about peace and safety.
"""
# The SemanticChunker is a text splitter that splits text into chunks based on semantic similarity. It takes the embeddings and breakpoint threshold type as input and returns a list of text chunks.

docs = text_splitter.create_documents([sample])

print("💭💭💭Number of chunks:", len(docs))
print("📖📖📖First chunk:", docs[0])

print("📚📚📚All chunks:", docs)

