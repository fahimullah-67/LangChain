from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", dimension=300)

documents = [
    "Artificial intelligence is transforming modern software development. Machine learning models can analyze data and make predictions with high accuracy.",
    "AI systems are widely used in applications like chatbots and recommendation engines. They rely on data and algorithms to learn patterns.",
    "Pizza is a popular fast food made with cheese, tomato sauce, and toppings. It is enjoyed all over the world.",
    "Football is one of the most popular sports globally. Teams compete to score goals and win matches.",
    "Python is a popular programming language used for web development, data science, and automation. It is known for its simplicity and readability.",
    "Cooking involves preparing meals using ingredients like vegetables, meat, and spices. Recipes help guide the process of making delicious food.",
    "Data science combines statistics, programming, and domain knowledge to extract insights from data. It often uses Python and machine learning techniques."
]

query = "AI"

embedding_documents = embedding.embed_documents(documents)
embedding_query = embedding.embed_query(query)

# print(embedding_documents)
# print(embedding_query)

similarity_scores = cosine_similarity([embedding_query], embedding_documents)[0]
top_match = sorted(enumerate(similarity_scores), key=lambda x:x[1])[-1]
print(top_match)

index, score = top_match

print(f"Query: {query}")
print(f"Top match: {documents[index]}")