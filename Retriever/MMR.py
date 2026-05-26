from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="Artificial Intelligence (AI) is the simulation of human intelligence in machines designed to think,learn, and solve problems. AI systems can perform tasks such as speech recognition, decision-making, and visual perception."),
    Document(page_content="AI has become one of the most transformative technologies of the modern era. From healthcare to finance, it is reshaping industries and creating new opportunities."),
    Document(page_content="The concept of AI dates back to the 1950s, but recent advances in computing power and data availability have accelerated its growth dramatically."),
    Document(page_content="AI is broadly categorized into narrow AI, which performs specific tasks, and general AI, which aims to replicate human-level intelligence."),
]

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store =  FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, 'lambda_mult':0}
)


query = "what is AI"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n-----Result {(i+1)} ----")
    print(doc.page_content)