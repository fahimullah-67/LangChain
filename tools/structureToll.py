from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class InputSchema(BaseModel):
    a:int = Field(required=True, description="The first number to add")
    b:int = Field(required=True, description="The second number to add")

def adding(a: int, b: int) -> int:
    return a + b

adding_tool = StructuredTool(
    name="adding",
    description="A tool for adding two numbers",
    func=adding,
    args_schema=InputSchema
)


result = adding_tool.invoke({"a": 1, "b": 2})

print(result)

print(adding_tool.name)
print(adding_tool.description)
print(adding_tool.args)