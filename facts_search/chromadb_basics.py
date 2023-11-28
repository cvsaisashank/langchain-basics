from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

#  Get the open_api_key from .env file
load_dotenv()

# Create an `embeddings` instace from Open AI
embeddings = OpenAIEmbeddings()

# This tells how to create chunks of the file. It first finds 200 characters and then finds the nearest `separator`.
text_splitter = CharacterTextSplitter(chunk_size=200, separator="\n", chunk_overlap=0)

loader = TextLoader("facts_search/facts.txt")

# find the file and extract all the content inside it. It outputs a List of Documents.
# docs = loader.load()

# find the file and extract all the content inside it into small chunks.Each chunk is a Document. It outputs a List of Documents.
docs = loader.load_and_split(text_splitter=text_splitter)

""" 
we create a chroma instance.
We are telling chroma to calculate the embeddings by reaching out to OpenAI for each of the `docs`
Then, store the calculated embeddings in the SQLLite databse which is going to be placed in `persist_directory` folder
"""
db = Chroma.from_documents(
    docs, embedding=embeddings, persist_directory="facts_search/emb"
)

# Do a simillarity search with our stored embeddings to find the ones most simillar to the users's question.
# k :- to get how many documents back in the output.
related_documents_with_score_result = db.similarity_search_with_score(
    "What is an interesting fact about the english language ?", k=2
)

related_documents_result = db.similarity_search(
    "What is an interesting fact about the english language ?"
)

for result in related_documents_result:
    print("\n")
    print("page_content", result.page_content)


# for result in related_documents_with_score_result:
#     print("\n")
#     print("page_content", result[0].page_content)
#     print("we get score if we use `similarity_search_with_score` ", result[1])

# for doc in docs:
#     print("doc.page_content", doc.page_content)
#     print("\n")
