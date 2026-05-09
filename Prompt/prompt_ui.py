from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = GoogleGenerativeAI(model="gemini-flash-latest")

st.header("ReSearch Tool/ Paper/ ")

user_input = st.text_input("Write Your prompt.")

if st.button("Summarize"):
    if user_input:
        result = model.invoke(user_input)
        st.write(result)
    else:
        result.warning("Please enter a prompt first.")
