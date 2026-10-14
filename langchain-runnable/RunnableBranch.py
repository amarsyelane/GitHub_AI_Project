from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

# load environment variables from .env file
load_dotenv()

# Initialize two different language models
model = ChatCohere(model="command-a-03-2025", temperature=0.7)

# prompts for generating a report and summarizing it
prompt  = PromptTemplate(
    template='Write a detailed report about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following {text}',
    input_variables=['text']
)

# Parse the output of the model as a string
parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt, model, parser)

# Create a RunnableBranch that checks the length of the report and if it's greater than 500 words, it summarizes the report using the second prompt and the same model, otherwise it returns the original report
# x is the output of the report_gen_chain, which is passed to the RunnableBranch as input. If the length of x is greater than 500 words, it runs the RunnableSequence with prompt2, model and parser, otherwise it runs the RunnablePassthrough which returns x as is.
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

# Create a final RunnableSequence that runs the report generation and then the branch chain
final_chain = RunnableSequence(report_gen_chain, branch_chain)

response = final_chain.invoke({"topic": "Datacenters on global warming"})

print("LLM Response for Report 🤖🤖🤖:", response)
print("Number of words in the report 🤖🤖🤖:", len(response.split()))

