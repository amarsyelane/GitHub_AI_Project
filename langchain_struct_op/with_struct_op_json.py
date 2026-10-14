from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

select_llm = ChatCohere(model="command-a-03-2025")

# Schema definition using Pydantic

# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model = select_llm.with_structured_output(json_schema)

result = structured_model.invoke("""𝐑𝐚𝐡𝐮𝐥 𝐆𝐚𝐧𝐝𝐡𝐢’𝐬 𝐟𝐨𝐫𝐞𝐢𝐠𝐧 𝐭𝐫𝐢𝐩 𝐦𝐨𝐧𝐞𝐲 𝐭𝐫𝐚𝐢𝐥. 💰🔍

𝟓𝟒 𝐟𝐨𝐫𝐞𝐢𝐠𝐧 𝐭𝐫𝐢𝐩𝐬. 𝟐𝟐 𝐲𝐞𝐚𝐫𝐬. ₹𝟔𝟎 𝐜𝐫𝐨𝐫𝐞 𝐞𝐬𝐭𝐢𝐦𝐚𝐭𝐞𝐝 𝐬𝐩𝐞𝐧𝐝. ₹𝟏𝟏 𝐜𝐫𝐨𝐫𝐞 𝐝𝐞𝐜𝐥𝐚𝐫𝐞𝐝 𝐢𝐧𝐜𝐨𝐦𝐞.

And Rahul Gandhi still refuses to explain who funds his luxury global lifestyle.

The travel trail is PUBLIC.
The money trail is MISSING.

* 54 traceable foreign trips across Europe, the US, Middle East and Asia
* One 2015 Southeast Asia tour cost ₹4.5 crore, over 5X his declared annual income that year
* Multiple foreign trips flagged by CRPF between 2024–25
* Oman trip in May 2026 surfaced with no public itinerary or disclosure

So who paid?

* Foreign entities? Then where are the FCRA approvals?
* Personal funds? Then where are the tax and forex disclosures?
* Someone else? Then the questions become even bigger.

𝐓𝐡𝐢𝐬 𝐢𝐬 𝐚 𝐭𝐫𝐚𝐧𝐬𝐩𝐚𝐫𝐞𝐧𝐜𝐲 𝐢𝐬𝐬𝐮𝐞 𝐚𝐧𝐝 𝐚 𝐦𝐚𝐭𝐭𝐞𝐫 𝐨𝐟 𝐧𝐚𝐭𝐢𝐨𝐧𝐚𝐥 𝐬𝐞𝐜𝐮𝐫𝐢𝐭𝐲. 𝐖𝐡𝐞𝐫𝐞 𝐢𝐬 𝐈𝐧𝐝𝐢𝐚’𝐬 𝐋𝐞𝐚𝐝𝐞𝐫 𝐨𝐟 𝐎𝐩𝐩𝐨𝐬𝐢𝐭𝐢𝐨𝐧 𝐭𝐫𝐚𝐯𝐞𝐥𝐥𝐢𝐧𝐠? 𝐖𝐡𝐨 𝐢𝐬 𝐡𝐞 𝐦𝐞𝐞𝐭𝐢𝐧𝐠? 𝐖𝐡𝐨’𝐬 𝐟𝐮𝐧𝐝𝐢𝐧𝐠 𝐭𝐡𝐞𝐬𝐞 𝐭𝐫𝐢𝐩𝐬?

54 trips.
Rs 60 crore spend.
A gap of 50 crores.

𝐎𝐧𝐞 𝐬𝐢𝐦𝐩𝐥𝐞 𝐪𝐮𝐞𝐬𝐭𝐢𝐨𝐧: #𝐖𝐡𝐨𝐅𝐮𝐧𝐝𝐬𝐑𝐚𝐡𝐮𝐥""")

print(f"🤖🤖🤖: {result}")