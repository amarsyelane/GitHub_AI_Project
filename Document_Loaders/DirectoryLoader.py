from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredFileLoader, UnstructuredURLLoader, PyPDFLoader, OnlinePDFLoader, CSVLoader, JSONLoader, PythonLoader
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatCohere(model="command-a-03-2025", temperature=1.5)

# DirectoryLoader is a document loader that loads documents from a directory. It takes the directory path and a glob pattern as input and returns a list of Document objects, where each Document object contains the content of a file in the directory that matches the glob pattern and its metadata.
loader = DirectoryLoader(
    path='documents',
    glob='**/*.pdf', 
    loader_cls=PyPDFLoader
)

# The load method of the DirectoryLoader class reads the content of the files in the directory that match the glob pattern and creates a Document object for each file. The content of the Document object is the text extracted from the file, and the metadata contains the file path.
#load_documents = loader.load()
lazy_load_documents = loader.lazy_load()  # This will return an iterator that loads documents lazily

'''# load
print("Length of documents loaded:", len(load_documents))
print("📖📖📖  Document content:", load_documents[0].page_content)'''

# lazy load
for doc in lazy_load_documents:
    print("📚📚📚  Document metadata :", doc.metadata)# Remove this break to load all documents