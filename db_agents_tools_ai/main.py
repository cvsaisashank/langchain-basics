from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from tools.sql import list_tables, run_query_tool

#  Get the open_api_key from .env file
load_dotenv()

# create an instance of Chat-LLM
chat = ChatOpenAI()

# list of tables
tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        # Use this `SystemMessage` schema when there is no `input_variables` or `templating` is needed. Use this when we are hardcoding a string.
        SystemMessage(
            content=f"You are an AI that has access to a SQLite database. \n {tables}"
        ),
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # human-input will be coming in `input` variable
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)

# list of Tools used
tools = [run_query_tool]

agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor("How many users are in the database?")
agent_executor("How many users have provided a shipping address ?")
