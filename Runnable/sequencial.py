from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template = PromptTemplate(
    template= "write a joke about {topic}",
    input_variables= {'topic'}
)

parser = StrOutputParser()


template1 = PromptTemplate(
    template = "Summarize the {topic}",
    input_variables= {"topic"}
)

template2 = PromptTemplate(
    template = "Explain the {topic}",
    input_variables= {"topic"}
)

chain = RunnableSequence(template, model, parser, template1, model, parser, template2, model, parser)


result = chain.invoke({"topic": "cricket"})


print(result)