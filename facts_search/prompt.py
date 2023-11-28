from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

# Create an `embeddings` instace from Open AI
embeddings = OpenAIEmbeddings()

""" 
we create a chroma instance.
We are telling chroma to calculate the embeddings by reaching out to OpenAI for each of the `docs`
"""
db = Chroma(embedding_function=embeddings, persist_directory="facts_search/emb")
