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
model = ChatCohere(model="command-a-03-2025", temperature=1.5)

# create a function that count the number of words in a text
def count_words(text: str) -> int:
    return len(text.split())

'''# convert the function to a RunnableLambda
count_words_runnable = RunnableLambda(count_words)

print(count_words_runnable.invoke("Hello world, this is a test."))'''

# Define prompts for generating a joke
prompt  = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Parse the output of the model as a string
parser = StrOutputParser()

# Create a RunnableSequence that generates a joke and then counts the number of words in the joke
joke_gen_chain = RunnableSequence(prompt, model, parser)

# Create a RunnableParallel that runs the joke generation and the word count in parallel
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(count_words)
    #"word_count": RunnableLambda(lambda x: len(x.split()))
})

# Create a final RunnableSequence that runs the joke generation and then the parallel chain
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

response = final_chain.invoke({"topic": "Relationships"})

print("LLM Response for Joke 🤖🤖🤖:", response['joke'])
print("Number of words in the joke 🤖🤖🤖:", response['word_count'])

