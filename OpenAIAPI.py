# %%
import os
from openai import OpenAI
# %% Open AI'dan bir API key al ve onu sisteme kaydettiğin adı (for ex "OPENAI-KEY") yaz 
client = OpenAI(api_key= os.environ.get("OPENAI-KEY"))

completion = client.chat.completions.create(
    messages=[{
        "role":"user",
        "content": "The coffee is extremely ..."
    }],
    model="gpt-3.5-turbo",
)

print(completion.choices[0].message)