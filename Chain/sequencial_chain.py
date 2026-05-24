from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template1 = PromptTemplate(
    template = "gave me a details description of {topic}",
    input_variables= {"topic"}
)

template2 = PromptTemplate (
    template = "Generate 5 summaries point of \n {topic}",
    input_variables= {"topic"}
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Unemployed in India"})

print(result)

chain.get_graph().print_ascii()