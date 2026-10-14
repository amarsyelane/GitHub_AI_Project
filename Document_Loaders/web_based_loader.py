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

url = "https://www.amazon.in/Samsung-Internal-Expansion-Graphics-MZ-V9P1T0B/dp/B0B9C3ZVHR?th=1"
loader = WebBaseLoader(url)

try:
    docs = loader.load()
    
    if not docs:
        print("Error: No documents were loaded from the URL.")
    else:
        parser = StrOutputParser()
        chain = prompt | llm | parser
        response = chain.invoke({"question": "What is the price of this product?", "text": docs[0].page_content})
        print(response)
except Exception as e:
    print(f"Error loading or processing the URL: {e}")
