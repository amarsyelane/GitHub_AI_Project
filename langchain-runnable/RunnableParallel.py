
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

 
# Define prompts for LinkedIn and Twitter
prompt_linkdin = PromptTemplate(
    template='Write a LinkedIn post about {topic}',
    input_variables=['topic']
)

prompt_twitter = PromptTemplate(
    template='Write a Twitter post about {topic}',  
    input_variables=['topic']
)

# Initialize a simple string output parser
parser = StrOutputParser()

# Create a parallel runnable that generates content for both LinkedIn and Twitter simultaneously
parallel_chain = RunnableParallel(
    linkdin = RunnableSequence(prompt_linkdin, model1, parser),
    twitter = RunnableSequence(prompt_twitter, model2, parser)
)

# Invoke the parallel chain with a topic and print the results
response = parallel_chain.invoke({"topic": "Artificial Intelligence"})

# Print the responses for both LinkedIn and Twitter
print("❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️ \n")
print("LLM Response for LinkedIn 🤖🤖🤖:", response['linkdin'])
print("💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕💕 \n")
print("LLM Response for Twitter 🤖🤖🤖:", response['twitter'])
print("❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️ \n")