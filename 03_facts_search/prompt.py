from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from redundant_filter_retriever import RedundantFilterRetriever

load_dotenv()

chat = ChatOpenAI()

# Create an `embeddings` instace from Open AI
embeddings = OpenAIEmbeddings()

""" 
we create a chroma instance.
We are telling chroma to calculate the embeddings by reaching out to OpenAI for each of the `docs`
Here we are not instantly creating the embedindings like we did in `Chroma.from_documents(docs, embedding=embeddings, persist_directory="facts_search/emb")`
"""
db = Chroma(embedding_function=embeddings, persist_directory="facts_search/emb")

# Below is a 'Chroma Retriever'
# retriever = db.as_retriever()

# Below is our custom retriever which will remove the duplicates
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)


chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.run("WHat is an interesting fact about the English language ?")

print("\n")
print(result)
print("\n")
