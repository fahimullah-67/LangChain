from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template = PromptTemplate(
    template = "gave me 3 fact about {topic}",
    input_variables= "{topic}"
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"topic":"cricket"})

print(result)