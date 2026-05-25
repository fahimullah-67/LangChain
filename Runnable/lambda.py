from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

def wordCount(text):
     return len(text.split())

template = PromptTemplate(
    template= "write a short joke about {topic}",
    input_variables= {'topic'}
)

parser = StrOutputParser()

runnableSequential = RunnableSequence(template, model, parser)

runnableParallel = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "wordCount" : RunnableLambda(wordCount)
}
)

finalRunnable = RunnableSequence(runnableSequential | runnableParallel)

result = finalRunnable.invoke({"topic":"Cricket"})

final_result = """ {} \n Word Count - {}""".format(result['joke'], result['wordCount']) 

print(final_result)


