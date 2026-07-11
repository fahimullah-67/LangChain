from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from lanchain_core.runnable import RunnableParallel, RunnablePassthrough, runnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2, max_output_tokens=1024)

parse = StrOutputParser()

video_id = "P26AE7NLx4Q"

api = YouTubeTranscriptApi()

try:
    transcript_list = api.fetch(video_id, languages=['en'])

    transcript = " ".join([chunk.text for chunk in transcript_list])
    print("Transcript fetched successfully. ->> ")
    # print(transcript)
except TranscriptsDisabled:
    print("Transcript is disabled for this video.")


splitter =  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(transcript)
print(f"Transcript split into {len(chunks)} chunks.")
# for i, chunk in enumerate(chunks[0:5]):  # Print the first 5 chunks
#     print(f"Chunk {i+1}: {chunk[:100]}...")  # Print the first 100 characters of each chunk


embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.from_texts(chunks, embedding)
print(f"Chunks Embedding successfully")
# print(vector_store.index.ntotal)
# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(['4']))


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# docs = retriever.invoke("What is the purpose of the given content?")

# for i, doc in enumerate(docs, 1):
#     print(f"\nDocument {i}:")
#     print(doc.page_content)


prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the question using only the provided context.
If the context does not contain the answer, reply with:
"I don't know."

Context:
{content}

Question:
{Question}
""",
    input_variables=["content", "Question"]
)

Question = "Taking about two people argument. what the argument between them."
retriever_docs = retriever.invoke(Question)
# print(retriever_docs)

def format_doc(retriever_docs):
    content_text = "\n\n".join(doc.page_content for doc in retriever_docs)
    return content_text

# content = "\n\n".join(doc.page_content for doc in retriever_docs)

# final_prompt = prompt.invoke({"content": content, "Question": Question})

parallel_chain = RunnableParallel({
    'contents' : retriever_docs | runnableLambda(format_doc),
    'Question' : RunnablePassthrough()
    }
    )

parallel_result = parallel_chain.invoke({"Question": Question})


chain = prompt | model | parse
final = chain.invoke(parallel_result)

print(final)









# import time

# for i in range(3):
#     try:
#         result = model.invoke(final_prompt)
#         print(result.content)
#         break
#     except Exception as e:
#         print(e)
#         print("Retrying...")
#         time.sleep(5)

# print(result.content)

