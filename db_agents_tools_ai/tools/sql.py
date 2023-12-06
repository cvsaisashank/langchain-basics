import sqlite3

from langchain.tools import Tool

# Make a connection to the SQLite database
conn = sqlite3.connect("db.sqlite")


# Define this function to send the list of tables info to `SystemMessage` in the `Prompt`` for ChatGpt to understand what tables are involved.
def list_tables():
    c = conn.cursor()
    # fetch the list of tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()

    # print in a readable way to send to the prompt and then to LLM
    return "\n".join(row[0] for row in rows if row[0] is not None)


# Define a function that is actually going to be executed whenver ChatGPT decided that it needs to execute a query.
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


# Setup Tool: Tool is going to take the configutation given and then Serialize to a `function object`
run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",  # this description is used by ChatGPT on when to run the tool.
    func=run_sqlite_query,  # Pass a function that needs to be executed whenever ChatGpt decides to runa function.
)
