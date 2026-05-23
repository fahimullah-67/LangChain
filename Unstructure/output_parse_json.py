from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

parse = JsonOutputParser()

template = PromptTemplate(
    template = 'Gave me the name, age and city of the fictional person \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction': parse.get_format_instructions()}
)

prompt = template.format()


chain = template | model | parse

final = chain.invoke({})

print(final)