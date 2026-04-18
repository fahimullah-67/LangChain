# # from langchain_openai import OpenAi
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv


# load_dotenv()

# # llm = OpenAi(model = "GPT-2.5")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
# )
# result = llm.invoke("What is Capital of Pakistan?")

# print(result)



from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Use the exact name from your 'models' list
llm = GoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)



try:
    result = llm.invoke("What is the Capital of Pakistan?")
    print(result)
    # print(result.content[0]["text"].replace("**", ""))
except Exception as e:
    print(f"Error: {e}")
