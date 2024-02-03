import langchain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from redundant_filter_retriever import RedundantFilterRetriever

# we use this here to understand what exactly the documents are returned by our custom retriver `RedundantFilterRetriever`
langchain.debug = True

load_dotenv()

chat = ChatOpenAI()

# Create an `embeddings` instace from Open AI
embeddings = OpenAIEmbeddings()

""" 
we create a chroma instance.
We are telling chroma to calculate the embeddings by reaching out to OpenAI for each of the `docs`
Here we are not instantly creating the embedindings like we did in `Chroma.from_documents(docs, embedding=embeddings, persist_directory="facts_search/emb")`
"""
db = Chroma(embedding_function=embeddings, persist_directory="emb")

# Below is a 'Chroma Retriever'. Commented this as we will use our custom_retriver to filter out duplicate
# documents before sending in the prompt. If we enable below code , we can see we are sending duplicate documents to the prompt.

# retriever = db.as_retriever()

# Below is our custom retriever which will remove the duplicate documents unlike `db.as_retriever()` which doesn not do.
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

# RetrievalQA Chain:
chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.run("What is an interesting fact about the English language ?")

print("\n")
print("The result is as below: \n", result)
print("\n")
