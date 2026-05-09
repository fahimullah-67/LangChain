from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

template1= PromptTemplate(
    template= "write a details report of {topic}",
    user_input = '{topic}' 
)

template2 = PromptTemplate(
    template ="write the 5 line summary of the following text /n {text}",
    user_input= "{text}"
)


prompt1 = template1.invoke({
    "topic":"black whole"
    })

result = model.invoke(prompt1)

report_text = result.content

prompt2 = template2.invoke({
    "text": report_text
})

result1 = model.invoke(prompt2)

print(result1.content[0]['text'])
