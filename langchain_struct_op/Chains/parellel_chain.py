from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, chain, chain

load_dotenv()

model1 = ChatCohere(model="command-a-03-2025")
model2 = ChatCohere(model="command-r7b-12-2024")

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain


text = """Quantum computing is an advanced computing technology that uses qubits, which can exist in multiple states at once through superposition and be interconnected via entanglement, allowing it to solve certain complex problems much faster than classical computers. Although still in an early stage and limited by technical challenges like error rates and scalability, it has strong future potential in areas such as cryptography (including breaking and developing secure encryption), drug discovery and healthcare through molecular simulation, optimization in logistics and finance, artificial intelligence, material science, and climate modeling. Over the coming decades, quantum computing is expected to complement classical systems by tackling highly complex tasks that are currently impractical or impossible to solve efficiently."""

result = chain.invoke({'text': text})
#print(f"🤖🤖🤖: {result}")

chain.get_graph().print_ascii()