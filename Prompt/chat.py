from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

chatHistory = [
    SystemMessage(content="You are a helpful AI assistant, give short answers")
]

print("Enter Your Query!...")

while True:
    user_text = input("User: ")

    if user_text.lower() == "exit":
        break

    user_message = HumanMessage(content=user_text)
    chatHistory.append(user_message)

    ai_response = model.invoke(chatHistory)

    print("AI:", ai_response.content[0]["text"])

    chatHistory.append(AIMessage(content=ai_response.content))

print(chatHistory)