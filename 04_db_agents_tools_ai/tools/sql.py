import sqlite3
from typing import List

from langchain.tools import Tool

# Langchain is stuck on Pydantic V1 so V2.BaseModel will not work as of now
from pydantic import BaseModel

# Make a connection to the SQLite database
conn = sqlite3.connect("db.sqlite")


# Define this function to get and then send the list of tables info to `SystemMessage` in the `Prompt` for ChatGpt to understand what tables are involved.
def list_tables():
    c = conn.cursor()
    # fetch the list of tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()

    # print the tables in a readable way to send to the prompt and then to LLM
    # Output-structure:
    # "users
    #  orders "

    return "\n".join(row[0] for row in rows if row[0] is not None)


# Define a function that is actually going to be executed whenver ChatGPT decided that it needs to execute a query.
# We will pass this function to a tool that needs to be executed whenever ChatGpt decides to run a function.
def run_sqlite_query(query):
    # This returns an object which gives us access to the database
    c = conn.cursor()
    try:
        # Execute the query
        c.execute(query)

        # We are collecting all the information for all the different rows that get returned, and we are sending
        # it back to ChatGPT by returning it from this funciton.
        return c.fetchall()

    except sqlite3.OperationalError as err:
        # The below return will directly be send back to ChatGPT
        return f"The following error occured: {str(err)}"


# Lanchain internally uses this information `query` as `str` to better describe the different arguments that
# ChatGPt or LLM should be providing to our tool.
# In other words, we are telling LLM or chatGPT to use our tool, you need to provide the below argument
class RunQueryArgsSchema(BaseModel):
    query: str


# Setup Tool: Tool is going to take the configutation given and then Serialize to a `function object` to send in the Prompt
run_query_tool = Tool.from_function(
    name="run_sqlite_query",  # tool name
    description="Run a sqlite query.",  # this description is used by ChatGPT on when to run the tool.
    func=run_sqlite_query,  # Pass a function that needs to be executed whenever ChatGpt decides to run a function.
    args_schema=RunQueryArgsSchema,  # Lanchain internally uses
)


# Define a function that is actually going to be executed whenver ChatGPT decided that it needs to get table schema.
def describe_tables(table_names):
    c = conn.cursor()

    tables = ", ".join(
        "'" + table + "'" for table in table_names
    )  # Output-structure: "'users', 'orders', 'products'"

    # Gives us back list of tuples  amd then first element inside the tuple is going to be the structure or schema of the chosen table
    rows = c.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});"
    )

    # print the table structure in a readable way to send to the prompt and then to LLM
    return "\n".join(row[0] for row in rows if row[0] is not None)


class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]


# Setup Tool: Tool is going to take the configutation given and then Serialize to a `function object` to send in the Prompt
describe_tables_tool = Tool.from_function(
    name="describe_tables",  # tool name
    description="Given a list of table names, returns the schema of those tables",  # this description is used by ChatGPT on when to run the tool.
    func=describe_tables,  # Pass a function that needs to be executed whenever ChatGpt decides to run a function.
    args_schema=DescribeTablesArgsSchema,
)
