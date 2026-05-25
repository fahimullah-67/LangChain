from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

loader = PyPDFLoader("AI.pdf")

docs = loader.load()

# print(len(docs))
# print(docs[0].page_content)

for document in docs:
    print(document.page_content)

print(type(docs))