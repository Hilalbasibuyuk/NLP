import openai
from openai import OpenAI
import pandas as pd
import os
openai.api_key = os.environ["OPENAI-KEY"]
from pprint import pprint

openai_client = OpenAI()

model_name = "gpt-3.5-turbo"

def chain_of_thought(user_query:str):
    system_role_definition =  "You are a helpful assistent and answer precise and concise."
    user_query_complete = f"{user_query}, think step-by-step."

    messages = [
        {"role": "system",
         "content": system_role_definition},
        {"role": "user", 
         "content": user_query_complete}
         ]
    response = openai_client.chat.completions.create(model=model_name, messages=messages)
    content = response.choices[0].message.content
    return content

user_prompt = "The goal of the Game of 24 is to use the four arithmetic operations (addition, subtraction, multiplication, and division) to combine four numbers and get a result of 24. The numbers are 3, 4, 6, and 8. It is mandatory to use all four numbers. Please check the final equation for correctness. Hints: Identify the basic operations, Prioritize multiplication and division, Look for combinations that make numbers divisible by 24, Consider order of operations, Use parentheses strategically, Practice with different number combinations"
res = chain_of_thought(user_query=user_prompt)
pprint(res)