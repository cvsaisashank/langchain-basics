import json

from langchain.tools import StructuredTool
from pydantic import BaseModel


def write_html_report(filename, html):
    with open(filename, "w") as f:
        f.write(html)


class WriteHtmlReportArgsSchema(BaseModel):
    filename: str
    html: str


# We use `StructuredTool` when we are dealing with multipel arguments
write_html_report_tool = StructuredTool.from_function(
    name="write_html_report",
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report",
    func=write_html_report,
    args_schema=WriteHtmlReportArgsSchema,
)


def write_json_report(filename, json_string):
    with open(filename, "w") as f:
        # Assuming json_string is a string in JSON format
        data = json.loads(json_string)
        json.dump(data, f, indent=4)


class WriteJsonReportArgsSchema(BaseModel):
    filename: str
    json_string: str


# We use `StructuredTool` when we are dealing with multiple arguments
write_json_report_tool = StructuredTool.from_function(
    name="write_json_report",
    description="Write an json file to disk. Use this tool whenever someone asks for a report. Input json should be a collection of key-value pairs.",
    func=write_json_report,
    args_schema=WriteJsonReportArgsSchema,
)
