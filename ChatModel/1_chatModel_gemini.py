from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI (
    model = "gemini-flash-latest",
    temperature= 1.5,
)
    # max_output_tokens= 10,

result = model.invoke("Write the 5 male name in pakistan" )
# result = model.invoke("What is the capital of Pakistan")
# print(result)
print(result.content[0]["text"].replace("**", ""))
