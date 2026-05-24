from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template1 = PromptTemplate(
    template = "gave me a details of {text}",
    input_variables= {"text"}
)

template2 = PromptTemplate(
    template = " generate quiz about {text}",
    input_variables= {"text"}
)

template3 = PromptTemplate(
    template = "merge the both note and quiz \n note -> {note}, quiz -> {quiz}",
    input_variables= {"note", "quiz"}
)

parser = StrOutputParser()

Parallel_Chain = RunnableParallel({
    "note" : template1 | model | parser,
    "quiz" : template2 | model | parser 
    }
)

merge_chain = template3 | model | parser

chain = Parallel_Chain | merge_chain

text = """
LangChain is a powerful framework designed to help developers build applications using large language models. It provides tools for chaining together different components like prompts, models, memory, and external data sources. With LangChain, developers can create chatbots, question-answering systems, and retrieval-augmented generation (RAG) pipelines more efficiently. It supports integration with APIs, databases, and vector stores, making it ideal for building intelligent applications. The framework emphasizes modularity and flexibility, allowing developers to customize workflows easily. Overall, LangChain simplifies the process of developing advanced AI-powered applications while maintaining scalability and performance.
"""

result = chain.invoke({"text": text})

print(result)
chain.get_graph().print_ascii()