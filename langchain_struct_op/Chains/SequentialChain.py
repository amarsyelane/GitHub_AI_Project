from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = ChatCohere(model="command-a-03-2025")

# first prompt -> Detail report
template_one = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"]
)


# second prompt -> Summary
template_two = PromptTemplate(
    template="Write a 5 lines summary of the following text: /n {text}",
    input_variables=["text"]
)

'''prompt_one = template_one.invoke({"topic": "The impact of climate change on global agriculture."})
result = llm.invoke(prompt_one)

prompt_one = template_two.invoke({"text": result.content})
result = llm.invoke(prompt_one)
print(f"🤖🤖🤖: {result.content}")'''

# stroutput parser
parser = StrOutputParser()
chain = template_one | llm | parser | template_two | llm | parser

result = chain.invoke({"topic": "Pakistan's begging for a loan"})
print(f"🤖🤖🤖: {result}")