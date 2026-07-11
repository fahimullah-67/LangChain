from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

class BaseModelTool(BaseTool):
    name : str = "base_model_tool"
    description : str = "A base model tool that adding two numbers"
    args_schema : Type[BaseModel] = InputSchema

    def _run(self, a: int, b: int) -> int:
        """Run the tool."""
        return a + b

adding_tool = BaseModelTool()

result = adding_tool.invoke({"a": 1, "b": 2})

print(result)