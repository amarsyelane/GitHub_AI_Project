

from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal

load_dotenv()

model1 = ChatCohere(model="command-a-03-2025")
model2 = ChatCohere(model="command-r7b-12-2024")

class FeedbackSentiment(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback")

parser_feedback = PydanticOutputParser(pydantic_object=FeedbackSentiment)

prompt = PromptTemplate(
    template='Classify the feedback text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback']
    ,partial_variables={'format_instructions': parser_feedback.get_format_instructions()}
)

parser = StrOutputParser()

# classifier chain
classifier_chain = prompt | model1 | parser_feedback

prompt_pos = PromptTemplate(
    template='Write a appropriate response to this positive feedback: \n {feedback}',
    input_variables=['feedback']    
)

prompt_neg = PromptTemplate(
    template='Write a appropriate response to this negative feedback: \n {feedback}',
    input_variables=['feedback']    
)

# branching chain
branching_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt_pos | model1 | parser),
    (lambda x: x.sentiment == 'negative', prompt_neg | model1 | parser),
    RunnableLambda(lambda x: "Invalid sentiment")
)

chain = classifier_chain | branching_chain

response = chain.invoke({"feedback": "The product quality is worst and I am not satisfied with my purchase!"})

print("LLM Response 🤖🤖🤖:", response)

# classifier_chain.get_graph().print_ascii()