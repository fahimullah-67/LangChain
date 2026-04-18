from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
# import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")

model1 = ChatHuggingFace(llm=llm)

result = model1.invoke("What is the capital of Pakistan?")

print(result.content)