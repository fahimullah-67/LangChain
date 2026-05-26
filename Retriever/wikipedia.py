import wikipedia
from langchain_community.retrievers import WikipediaRetriever

wikipedia.set_lang("en") 

wiki_retriever = WikipediaRetriever(top_k_results=2, language="en")

query = "the geopolitical history of pakistan and india from the perspective of china."

docs = wiki_retriever.invoke(query)

for doc in docs:
    print(doc.page_content)