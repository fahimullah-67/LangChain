from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate([
    ("system", 'your helpful customer support'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
]) 


chat_history =[]
with open('chat_history.txt') as f:
    chat_history.extend(f.readline())

# print(chat_history)

prompt = prompt_template.invoke({'chat_history':chat_history, 'query':'where is my refound'})

print(prompt)

