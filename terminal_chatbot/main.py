# import argparse

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    FileChatMessageHistory,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

#  Get the open_api_key from .env file
load_dotenv()

# create an instance of Chat-LLM
chat = ChatOpenAI()

# memory_key :- `Memory` adds a new key to the dictionary on top of the input's that are coming in.
# return_messages :- It will intelligently convert the strings to intelligent objects like `HumanMessage("What is 1 + 1?", AIMessage("2")`)
memory1 = ConversationBufferMemory(
    memory_key="messages_history",
    return_messages=True,
    chat_memory=FileChatMessageHistory("terminal_chatbot/messages.json"),
)

# use either memory1 or memory2. memory2 saves cost by summarizing the prompt.
memory2 = ConversationSummaryMemory(
    memory_key="messages_history",
    return_messages=True,
    llm=chat,
    # chat_memory=FileChatMessageHistory("terminal_chatbot/messages1.json"),
)

# use this class to create a Prompt with all the history of messages and the latest human asked question.
prompt = ChatPromptTemplate(
    input_variables=["content", "messages_history"],
    messages=[
        MessagesPlaceholder(variable_name="messages_history"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

# Create a Chain by passing the necessary inputs which in this case are `llm`, `prompt`, `memory`
chain = LLMChain(llm=chat, prompt=prompt, memory=memory2, verbose=True)

# Run an infinite loop to ask follow up question in the terminal.
while True:
    content = input(">>")
    result = chain({"content": content})
    print(result["text"])
