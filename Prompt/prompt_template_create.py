from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
    template="""
    You are a research assistant. Summarize the following academic paper content.

    Explanation Style: {style_input}
    Target Length: {length_input}

    Paper Content: {paper_input}
    """,
    input_variables=['paper_input','style_input','length_input'],
    validate_template= True 
)

template.save("prompt.json")