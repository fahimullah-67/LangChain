
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda 
from pydantic import BaseModel, Field 
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

class feedback(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="Gave the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=feedback)

template1 = PromptTemplate(
    template = "Classified a sentiment of the following feedback text into positive or negative. {feedback} \n {{formate_instructions}}",
    input_variables= {"feedback"},
    partial_variables= {"formate_instructions":parser2.get_format_instructions()}
)

parser= StrOutputParser()

ClassifierChain = template1 | model | parser2

# result1 = ClassifierChain.invoke({"feedback": "this is beautiful Phone"})

# print(result1.sentiment)
# print(result1)

template2 = PromptTemplate(
    template="""
    Write a SHORT and direct thank-you response for this positive feedback.

    DO NOT give multiple options.
    DO NOT explain anything.
    ONLY give one final response.

    Feedback:
{feedback}
""",
    input_variables=["feedback"]
)

template3 = PromptTemplate(
    template="""
    Write a SHORT professional apology response for this negative feedback.

    DO NOT give multiple templates.
    DO NOT explain anything.
    ONLY give one response.

    Feedback:
    {feedback}
""",
    input_variables=["feedback"]
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', template2 | model | parser),
    (lambda x:x.sentiment == 'negative', template3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = ClassifierChain | branch_chain

# text = """
# This is a terrible Smart Phone
# """

result = chain.invoke({"feedback" : "This is a terrible Smart Phone"})

print(result)

chain.get_graph().print_ascii()