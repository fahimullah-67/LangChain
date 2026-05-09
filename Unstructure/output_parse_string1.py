from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template1= PromptTemplate(
    template= "write a details report of {topic}",
    user_variable = '{topic}' 
)

template2 = PromptTemplate(
    template ="write the 5 line summary of the following text /n {text}",
    user_variable= "{text}"
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({
    'topic':'black hole'
})

print(result)


