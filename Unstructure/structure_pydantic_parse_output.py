from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class Facts(BaseModel):
    fact_1: str = Field(description="fact 1 about the topic")
    fact_2: str = Field(description="fact 2 about the topic")
    fact_3: str = Field(description="fact 3 about the topic")

parser = JsonOutputParser(pydantic_object=Facts)

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

prompt = PromptTemplate(
    template="Give three facts about {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

result = chain.invoke({"topic": "black hole"})
print(result)