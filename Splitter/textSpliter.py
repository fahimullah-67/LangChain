from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("AI.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=" "
)

text = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines designed to think, learn,
and solve problems. AI systems can perform tasks such as speech recognition, decision-making, and
visual perception.

AI has become one of the most transformative technologies of the modern era. From healthcare to
finance, it is reshaping industries and creating new opportunities.

The concept of AI dates back to the 1950s, but recent advances in computing power and data
availability have accelerated its growth dramatically.

AI is broadly categorized into narrow AI, which performs specific tasks, and general AI, which aims to
replicate human-level intelligence.
"""

result = splitter.split_documents(docs)

print(result[0].page_content)