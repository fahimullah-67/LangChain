from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

documents = [
    "the capital of France is Paris",
    "the capital of Germany is Berlin",
    "the capital of Pakistan is Islamabad",
]

result = embedding.embed_documents(documents)

print(result)
