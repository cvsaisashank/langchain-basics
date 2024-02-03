from dotenv import load_dotenv
from handlers.chat_model_start_handler import ChatModelStartHandler
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from tools.report import write_html_report_tool, write_json_report_tool
from tools.sql import describe_tables_tool, list_tables, run_query_tool

#  Get the open_api_key from .env file
load_dotenv()

# use this handler-callbacks to debug what is being sent to LLM
handler = ChatModelStartHandler()

# create an instance of Chat-LLM. Pass the necessary callback-handlers to debug what is being sent from langchain to LLM.
chat = ChatOpenAI(callbacks=[handler])

# list of tables
tables = list_tables()

print("list of tables - this is printed by me", tables)

prompt = ChatPromptTemplate(
    messages=[
        # Use this `SystemMessage` schema when there is no `input_variables` or `templating` is needed. Use this when we are hardcoding a string.
        SystemMessage(
            content=(
                "You are an AI that has access to a SQLite database."
                f"The database has tables of: {tables} \n"
                "Do not make any assumptions about what tables exist "
                "or what columsn exist. Instead, use the 'describe_tables' function."
                "Always run 'describe_tables' function first before running 'run_sqlite_query' function"
                "For HTML report, use the 'write_html_report' function"
                "For JSON report, use the 'write_json_report' function"
            )
        ),
        # we place the memory here before a brand new human input comes in.
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # human-input will be coming in `input` variable
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)

# we will add memory to agent-execetor to retain the conversation history
# memory_key :- `Memory` adds a new key to the dictionary on top of the input's that are coming in.
# return_messages :- It will intelligently convert the strings to intelligent objects like `HumanMessage("What is 1 + 1?", AIMessage("2")`)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# list of Tools used
tools = [
    run_query_tool,
    describe_tables_tool,
    write_html_report_tool,
    write_json_report_tool,
]

agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    # verbose=True
)

# RUN Any one of the below agent executor as ther eis a rate limitter on my free plan

# agent_executor("How many users are in the database?")
# agent_executor("How many users have provided a shipping address ?")
agent_executor(
    "Summarize the top 5 most popular products. Write the results to HTML report file ?"
)

# agent_executor("How many orders are there?")

# agent_executor("Repeat the same process for users?")
