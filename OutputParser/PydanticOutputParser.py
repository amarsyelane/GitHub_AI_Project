from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatCohere(model="command-a-03-2025")

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18,description="The age of the person")
    city: str = Field(description="The city where the person lives")    

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate a fictional person's name, age and city of {place}: \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

'''prompt = template.invoke({'place': 'France'})
response = llm.invoke(prompt)
result = parser.parse(response.content)'''

chain = template | llm | parser
result = chain.invoke({"place": "russia"})

#print("💭💭💭💭",prompt,"💭💭💭💭")
print(f"LLM Response🤖🤖🤖: {result}")
