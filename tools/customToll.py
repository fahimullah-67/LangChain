from langchain_core.tools import tool

def adding(a, b):
    """Adding two numbers."""
    return a + b

def adding(a: int, b: int) -> int:
    """Adding two numbers."""
    return a + b

@tool
def adding(a: int, b: int) -> int:
    """Adding two numbers."""
    return a + b

result = adding.invoke({"a": 1, "b": 2})
print(result)

print(adding.name)
print(adding.description)
print(adding.args)

print(adding.args_schema.model_json_schema())