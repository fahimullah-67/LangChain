from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

def wordCount(text):
     return len(text.split())

template = PromptTemplate(
    template= "Explain the {topic}",
    input_variables= {'topic'}
)

template1 = PromptTemplate(
    template= "summarize the {topic}",
    input_variables= {'topic'}
)

parser = StrOutputParser()

runnableSequence = RunnableSequence(template, model, parser)

runnableBranch = RunnableBranch(
    (lambda x: len(x.split())>200, RunnableSequence(template1, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(runnableSequence, runnableBranch)
result = chain.invoke({"topic":"Pakistan vs India"})

print(result)
