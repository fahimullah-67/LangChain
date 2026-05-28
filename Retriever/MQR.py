from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS



docs = [
    Document(page_content="Artificial Intelligence allows machines to mimic human thinking and decision making."),
    Document(page_content="Intelligent systems can learn patterns from data and improve over time without explicit programming."),
    Document(page_content="Machines today are becoming smarter, capable of reasoning and problem solving similar to humans."),
    Document(page_content="Machine learning and deep learning are techniques used to build predictive models."),
    Document(page_content="AI is widely used in cricket analytics to predict player performance and match outcomes."),
    Document(page_content="Cricket is played with a bat and ball between two teams of eleven players."),
    Document(page_content="Pakistan is located in South Asia and shares borders with India, China, and Afghanistan."),
    Document(page_content="Historical data plays an important role in training modern intelligent systems."),
    Document(page_content="Bananas contain potassium and are beneficial for heart health."),
    Document(page_content="Cats can sleep up to 16 hours a day depending on their environment."),
    Document(page_content="Cloud computing enables scalable storage and processing over the internet."),
    Document(page_content="Algorithms that adapt from experience are transforming industries globally."),
    Document(page_content="Systems that can understand language, images, and speech are shaping the future of technology."),
    Document(page_content="Data science combines statistics, programming, and domain knowledge to extract insights."),
    Document(page_content="Robots performing tasks in factories are not always intelligent systems."),
]

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

vector_store =  FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)


similarity_search = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

multipleQuery_retriever = MultiQueryRetriever.from_llm(
    retriever= vector_store.as_retriever(search_kwargs={"k":3}),
    llm=model, 
)

query = "smart machines"

similarity_results = similarity_search.invoke(query)
multiple_results = multipleQuery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n-----Similarity Search Result {(i+1)} ----")
    print(doc.page_content)

print("*"*50)

for i, doc in enumerate(multiple_results):
    print(f"\n-----Multiple Query Result {(i+1)} ----")
    print(doc.page_content)


