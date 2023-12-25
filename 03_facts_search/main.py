from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

#  Get the open_api_key from .env file
load_dotenv()

# create an instance of OpenAiEmbeddings
embeddings = OpenAIEmbeddings()
# This tells how to create chunks of the file. It first finds 200 characters and then finds the nearest `separator`.
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

loader = TextLoader("facts.txt")
# find the file and extract all the content inside it in chunks. It outputs a List of Documents.
docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

results = db.similarity_search(
    "What is an interesting fact about the English language?"
)

for result in results:
    print("\n")
    print(result.page_content)
