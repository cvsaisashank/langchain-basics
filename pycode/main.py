import argparse

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Get the open_api_key from .env file
load_dotenv()

parser = argparse.ArgumentParser()
# we need to tell the arser we will 2 command line arguments
parser.add_argument("--task", default="return a list of arguments")
parser.add_argument("--language", default="python")
# Get the args passed from command line
args = parser.parse_args()

# Below is a completion model.
llm = OpenAI()

# Below is a keyword argument in python
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}",
)

unit_test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a  unit test for the following {language} code:\n {code}",
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

unit_test_chain = LLMChain(llm=llm, prompt=unit_test_prompt, output_key="test")

chain = SequentialChain(
    chains=[code_chain, unit_test_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"],
)

result = chain({"language": args.language, "task": args.task})

print(">>>>> code")
print(result["code"])

print(">>>>> test")
print(result["test"])
