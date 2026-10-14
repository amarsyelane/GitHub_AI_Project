from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

select_llm = ChatCohere(model="command-a-03-2025")

# Schema definition using Pydantic

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = select_llm.with_structured_output(Review)

result = structured_model.invoke("""Very prompt staff.
Good co-ordination.
Great service.
Mr shekhar deshpande sir gives best advice.he checks car carefully and suggest advice.""")

print(f"🤖🤖🤖: {result}")