# RecursiveCharacterTextSplitter    
from langchain_text_splitters import Language, Language, RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredFileLoader, UnstructuredURLLoader, PyPDFLoader, OnlinePDFLoader, CSVLoader, JSONLoader, PythonLoader
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the Cohere model
model = ChatCohere(model="command-a-03-2025", temperature=0.7)


splitter = RecursiveCharacterTextSplitter.from_language(chunk_size=50, chunk_overlap=0,language=Language.PYTHON)

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
# The CharacterTextSplitter is a text splitter that splits text into chunks based on a specified character. It takes the chunk size, chunk overlap, and separator as input and returns a list of text chunks. The chunk size determines the maximum length of each chunk, the chunk overlap determines how much overlap there is between consecutive chunks, and the separator determines the character used to split the text.
result = splitter.split_text(PYTHON_CODE)

print("💭💭💭Number of chunks:", len(result))
print("📖📖📖First chunk:", result[0])

print("📚📚📚All chunks:", result)

