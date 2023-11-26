# import argparse

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# # Get the open_api_key from .env file
load_dotenv()

chat = ChatOpenAI( verbose=True)
# memory_key :- `Memory` adds a new key to the dictionary on top of the input's that are coming in.
# return_messages :- It will intelligently convert the strings to intelligent objects like `HumanMessage("What is 1 + 1?", AIMessage("2")`)
memory = ConversationBufferMemory(memory_key="messages_history", return_messages=True)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages_history"],
    messages=[
        MessagesPlaceholder(variable_name="messages_history"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">>")
    result = chain({"content": content})
    print(result["text"])
