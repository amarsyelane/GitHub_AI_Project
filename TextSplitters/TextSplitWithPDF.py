from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredFileLoader, UnstructuredURLLoader, PyPDFLoader, OnlinePDFLoader, CSVLoader, JSONLoader, PythonLoader
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the Cohere model
model = ChatCohere(model="command-a-03-2025", temperature=0.7)

# Load the PDF document using PyPDFLoader
loader = PyPDFLoader("Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron.pdf")

# The load method of the PyPDFLoader class reads the content of the PDF file and creates a Document object for each page in the file. The content of the Document object is the text extracted from the page, and the metadata contains the file path and the page number.
documents = loader.load()

# The CharacterTextSplitter is a text splitter that splits text into chunks based on a specified character. It takes the chunk size, chunk overlap, and separator as input and returns a list of text chunks. The chunk size determines the maximum length of each chunk, the chunk overlap determines how much overlap there is between consecutive chunks, and the separator determines the character used to split the text.
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=400, separator="")

result = splitter.split_documents(documents)

print("💭💭💭Number of chunks:", len(result))
print("📖📖📖First chunk:", result[0])

