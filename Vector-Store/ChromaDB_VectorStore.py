# references: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma

from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_cohere import CohereEmbeddings

load_dotenv()

# embeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Initialize the Cohere model
model = ChatCohere(model="command-a-03-2025", temperature=0)

'''# document objects to be added to the vector store
doc1 = Document(
    page_content="Pichai Sundararajan, better known as Sundar Pichai, is an Indian–American business executive who has been the CEO of Google since 2015 and the CEO of its parent company Alphabet Inc. since 2019. Pichai began his career as a materials engineer",
    metadata={"company": "Google", "position": "CEO"}
)

doc2 = Document(
    page_content="Satya Narayana Nadella is an Indian-American business executive. He is the chairman and chief executive officer (CEO) of Microsoft. Nadella joined Microsoft in 1992, and quickly rose through the ranks. He was named CEO in February 2014, succeeding Steve Ballmer.",
    metadata={"company": "Microsoft", "position": "CEO"}
)

doc3 = Document(
    page_content="Shantanu Narayen is an Indian-American business executive. He is the chairman, president and chief executive officer (CEO) of Adobe Inc. Narayen joined Adobe in 1998 as a senior vice president of worldwide product development. He was named CEO in December 2007, succeeding Bruce Chizen.",
    metadata={"company": "Adobe", "position": "CEO"}
)

doc4 = Document(
    page_content="Arvind Krishna is an Indian-American business executive. He is the chairman and chief executive officer (CEO) of IBM. Krishna joined IBM in 1990, and quickly rose through the ranks. He was named CEO in April 2020, succeeding Ginni Rometty.",
    metadata={"company": "IBM", "position": "CEO"}
)

doc5 = Document(
    page_content="Elon Reeve Musk is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of May 2026, Forbes estimates his net worth to be US$788 billion. He has also been involved in several other ventures, including Neuralink and The Boring Company.",
    metadata={"company": "Tesla", "position": "CEO"}
)

# Create a document list
documents = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# add documents to the vector store
vector_store.add_documents(documents)
'''

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
# view documents in the vector store
view_documents = vector_store.get(include=['metadatas', 'documents', 'embeddings'])
print("📚📚📚Documents in the vector store:", view_documents)

# search documents in the vector store
query = "Who is the CEO of Microsoft?"
results = vector_store.similarity_search(query, k=2)
print("🔍🔍🔍Search results for query:", results)

# search with similarity score
results_with_scores = vector_store.similarity_search_with_score(query, k=2)
print("🔍🔍🔍Search results with scores for query:", results_with_scores)

# metadata filtering
metadata_filter = {"company": "Microsoft"}
filtered_results = vector_store.similarity_search(query, k=2, filter=metadata_filter)
print("🔍🔍🔍Search results with metadata filter for query:", filtered_results)

# update documents in the vector store
updated_doc2 = Document(
    page_content="The paper endeavours to analyze authentic dimensions of Leadership with Satya Nadella's leadership and his aggressive strategy leading to an increase in Microsoft's profitability. A total of 7 interviews of Nadella from 2014 to 2020 were coded in NVivo to prepare Word query, TreeMap and cluster analysis. Also, besides tweets associated with #Satya Nadella were captured using NCapture and exported to NVivo to obtain the sentiments by performingsentiment analysis",
    metadata={"company": "Microsoft", "position": "CEO"}
)

vector_store.update_document(document_id=['ae37aa9c-f2d6-4bf4-9476-372249d2df8f'], document=[updated_doc2])

'''# delete a document from the vector store
vector_store.delete(ids=['0daf8242-e83a-4610-b518-ede0bad6c27a'])'''

# view updated documents in the vector store
updated_view_documents = vector_store.get(include=['metadatas', 'documents', 'embeddings'])
print("📚📚📚Updated documents in the vector store:", updated_view_documents)