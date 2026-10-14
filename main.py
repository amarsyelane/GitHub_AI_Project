# This is a sample code to demonstrate the usage of DuckDuckGoSearchRun tool from langchain_community and ChatCohere from langchain_cohere.
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

load_dotenv()

# Initialize the ChatCohere model with specified parameters
llm = ChatCohere(model="command-a-03-2025", temperature=0.7)

# Initialize the DuckDuckGoSearchRun tool
search = DuckDuckGoSearchRun()

result = search.invoke("Top news about AI in the last 24 hours")

print("🤖🤖Search Result:", result)

# References: https://docs.langchain.com/oss/python/integrations/tools/ddg
