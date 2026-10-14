from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatCohere(model="command-a-03-2025", temperature=1.0)

prompt = PromptTemplate(
    template='Answer the following question: {question} based on the following text: {text}',
    input_variables=['question','text']
)

loader = CSVLoader(file_path='genz_social_media_behavior.csv')

parser = StrOutputParser()

chain = prompt | llm | parser

data = loader.load()

response = chain.invoke({"question": "What are the key insights about Gen Z's social media behavior?", "text": data[0].page_content})

print("📖📖📖  Document content:", data[0].page_content)

print("📚📚📚  Document metadata :", data[0].metadata)

print("🤖🤖🤖LLM Response:", response)