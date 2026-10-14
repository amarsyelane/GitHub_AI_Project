
from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal

load_dotenv()

model1 = ChatCohere(model="command-a-03-2025")

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt_for_explanation = PromptTemplate(
    template='Explain the joke: {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model1, parser, prompt_for_explanation, model1, parser)

response = chain.invoke({"topic": "Women"})

print("LLM Response 🤖🤖🤖:", response)