from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

#  Get the open_api_key from .env file
load_dotenv()

embeddings = OpenAIEmbeddings()

# This tells how to create chunks of the file. It first finds 200 characters and then finds the nearest `separator`.
text_splitter = CharacterTextSplitter(chunk_size=200, separator="\n", chunk_overlap=0)

loader = TextLoader("facts_search/facts.txt")

# find the file and extract all the content inside it. It outputs a List of Documents.
# docs = loader.load()

# find the file and extract all the content inside it in chunks. It outputs a List of Documents.
docs = loader.load_and_split(text_splitter=text_splitter)

for doc in docs:
    print("doc.page_content", doc.page_content)
    print("\n")
