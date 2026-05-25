from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template = PromptTemplate(
    template= "write a joke about {topic}",
    input_variables= {'topic'}
)

template1 = PromptTemplate(
    template = "Explain the {topic}",
    input_variables= {"topic"}
)

parser = StrOutputParser()

runnableSequential = RunnableSequence(template, model, parser)

runnableParallel = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explanation" : RunnableSequence(template1, model, parser) 

    }
)

finalRunnable = runnableSequential | runnableParallel

print(finalRunnable.invoke({"topic": "cricket"}))

