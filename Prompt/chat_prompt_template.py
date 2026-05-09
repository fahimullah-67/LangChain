
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate([
    ('system', 'your are helpful {domain} expert'),
    ('human','tell in simple term about {topic} ')
])

result = prompt.invoke({'domain':'cricket', 'topic':'dusra'})

print(result)
