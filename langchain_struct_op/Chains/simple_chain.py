from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = ChatCohere(model="command-a-03-2025")

prompt = PromptTemplate(
    template="What are the advantages of {topic}?",
    input_variables=["topic"]
) 

parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"topic": "Python programming"})

print(f"LLM Response🤖🤖🤖: {result}")

chain.get_graph().print_ascii()