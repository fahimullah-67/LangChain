from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-flash-latest")


json_format = {
    "title": "ReviewSchema",
    "type": "object",
    "properties": {

        "key_theme": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down the key themes from review in list"
        },

        "summary": {
            "type": "string",
            "description": "Summarize the review"
        },

        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neu"],
            "description": "Return review sentiment"
        },

        "performance": {
            "type": "string",
            "description": "Summarize performance from review"
        },

        "pros": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down pros from review"
        },

        "cons": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down cons from review"
        },

        "name": {
            "type": "string",
            "description": "Return reviewer name"
        }
    },

    "required": [
        "key_theme",
        "summary",
        "sentiment"
    ]
}




structure_model = model.with_structured_output(json_format)

result = structure_model.invoke("""
Overview

After using this product for a while, the overall experience has been quite balanced. The setup process was simple, the interface was user-friendly, and most features worked as expected. It performs consistently for everyday use and gives a premium feel compared to many similar products in the same category.

Performance

The speed and responsiveness are impressive during normal usage. Tasks are completed efficiently, and the product maintains stable performance over time. In heavy usage situations, there can be occasional slowdowns, but nothing serious enough to affect regular productivity.

Design & Build

The design is modern and clean. Materials feel durable, and the overall appearance gives a professional look. The product is comfortable to use for long periods, though a few design elements may require some adjustment for new users.

Pros
Easy to use and beginner friendly
Smooth overall performance
Attractive and modern design
Reliable for daily usage
Good feature availability
Cons
Some features could be improved
Occasional lag during intensive tasks
Limited customization options
Price may feel slightly high for some users
Final Summary

Overall, the product provides a dependable and satisfying experience with a strong mix of usability, design, and performance. While there are a few areas that could be refined, it still stands out as a practical option for users looking for something reliable and efficient for everyday needs.

review by Fahim ullah
""")

print(result)
