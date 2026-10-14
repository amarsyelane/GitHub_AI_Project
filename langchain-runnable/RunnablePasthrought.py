from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal

# load environment variables from .env file
load_dotenv()

# Initialize two different language models
model1 = ChatCohere(model="command-a-03-2025")
model2 = ChatCohere(model="command-r7b-12-2024")

# Initialize a passthrough runnable
passthrough = RunnablePassthrough()

# Define prompts for generating a joke
prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Define a prompt for explaining the joke
prompt_for_explanation = PromptTemplate(
    template='Explain the joke: {text}',
    input_variables=['text']
)


parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model1, parser)

parallel_chain = RunnableParallel({    
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(passthrough, prompt_for_explanation, model1, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

response = final_chain.invoke({"topic": "Science"})

print("LLM Response for Joke 🤖🤖🤖:", response['joke'])
print("LLM Response for Explanation 🤖🤖🤖:", response['explanation'])

