
#langchain_community_document_loaders is a collection of document loaders contributed by the community. These document loaders are not maintained by the core LangChain team, but they can be useful for loading documents from various sources.

from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredFileLoader, UnstructuredURLLoader, PyPDFLoader, OnlinePDFLoader, CSVLoader, JSONLoader, PythonLoader
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatCohere(model="command-a-03-2025", temperature=1.5)


# PyPDFLoader is a document loader that loads documents from a PDF file. It takes the file path as input and returns a list of Document objects, where each Document object contains the content of a page in the PDF file and its metadata.

loader = PyPDFLoader("AgenticAI.pdf")

# The load method of the PyPDFLoader class reads the content of the PDF file and creates a Document object for each page in the file. The content of the Document object is the text extracted from the page, and the metadata contains the file path and the page number.
documents = loader.load()

# Prompt for summarizing the content of the documents
prompt = PromptTemplate(
    template='Summarize the following content: {text}',
    input_variables=['text']
)

# Parse the output of the model as a string
parser = StrOutputParser()

# Create a RunnableSequence that summarizes the content of the documents using the prompt and the model
chain = prompt | model | parser

# Invoke the chain with the content of the first document as input
response = chain.invoke({"text": documents[0].page_content})
print("📖📖📖  Document content:", documents[0].page_content)
print("📚📚📚  Document metadata :", documents[0].metadata)
print("🤖🤖🤖  LLM Response for Summary :", response)
