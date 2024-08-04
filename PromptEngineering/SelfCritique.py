from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_experimental.smart_llm import SmartLLMChain

hard_question = "If it takes 5 workers 8 days to build a house, how long would it take 10 workers to build the same house?"

prompt = PromptTemplate.from_template(hard_question)
llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo")

chain = SmartLLMChain(llm = llm, prompt = prompt, n_ideas = 3, verbose = True) # 3 fikirden en doğrusunu alır

chain.run({})