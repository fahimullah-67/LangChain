from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = GoogleGenerativeAI(model="gemini-flash-latest")


st.header("ReSearch Tool/ Paper/ ")

paper_input = st.selectbox("Select Your paper:", ["attention all you need", "BERT: pre training of  deep Bidirectionally", "GPT-3 language model are few shot Learner", "discuss model beat gain image"] )

style_input = st.selectbox("Select Summary Style:", ["Technical/Professional", "Simple/Layman", "Bullet Points"])

length_input = st.selectbox("Select Length:", ["Short (1 paragraph)", "Medium (3-4 paragraphs)", "Detailed Analysis"])


# prompt
template = load_prompt('prompt.json')

# fill the placeholder 



if st.button("Summarize"):
    chain = template | model

    result = chain.invoke({
    'paper_input': paper_input,
    'style_input':style_input,
    'length_input':length_input,
    })
    # prompt = template.invoke()  

    # result = model.invoke(prompt)
    # st.write("Hello Boss")
    st.write(result)
    print("Hello Boss")


