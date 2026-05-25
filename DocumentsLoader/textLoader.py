from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

parser = StrOutputParser()

template = PromptTemplate(
    template = "Summarize the poem \n {poem} ",
    input_variables = {"poem"}
)

loader = TextLoader("cricket.txt")

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = template | model | parser

print(chain.invoke({"poem": docs[0].page_content}))
