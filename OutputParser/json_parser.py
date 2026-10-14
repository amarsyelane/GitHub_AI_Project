from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm = ChatCohere(model="command-a-03-2025")
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me pros about: {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

'''prompt = template.format()
llm_response = llm.invoke(prompt)
result = parser.parse(llm_response.content)
'''

# chain helping to connect the prompt, llm and parser
chain = template | llm | parser
result = chain.invoke({"topic": "Python programming"})

print(f"LLM Response🤖🤖🤖: {result}")