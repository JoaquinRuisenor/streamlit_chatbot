import os
import sys

import pandas as pd
import tiktoken

from dotenv import load_dotenv

from langchain.vectorstores import Pinecone
import pinecone
import cohere


from langchain.embeddings.openai import OpenAIEmbeddings
import openai

from tools import cybersecurity_policy_pdf_parser, python_agent, csv_agent
from langchain_experimental.tools import PythonREPLTool


from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from langchain.tools.json.tool import JsonSpec
import json
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.agents import create_json_agent
from langchain.llms import OpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory




load_dotenv()
pinecone.init(api_key="977b78cf-6233-481d-bb18-b88ec09fb444", environment="gcp-starter")
co = cohere.Client(os.environ["COHERE_API_KEY"])

chat = ChatOpenAI(
    openai_api_key = os.environ["OPENAI_API_KEY"],
    # model = 'gpt-3.5-turbo'
    model = 'gpt-4'
)

pdf_path_input = 'C:\\Users\\JoaquinRuiseñor\\Documents\\OutThink DataScience\\LLM_Github\\CybersecurityDocs\\OutThink - Information Security Policy.pdf'
json_path_input = 'C:\\Users\\JoaquinRuiseñor\\Documents\\OutThink DataScience\\LLM_Github\\chatbot-langchain\\training.json'
# csv_path_input = 'C:\\Users\\JoaquinRuiseñor\\Documents\\OutThink DataScience\\LLM_Github\\chatbot-langchain\\training.csv'


tools = [
    Tool(
        name = "Cybsersecurity Policy Document QA system",
        func=cybersecurity_policy_pdf_parser(pdf_path_input).run,
        description="useful for when you need to answer questions about Cybersecurity polcies and compliance."
    ),
    Tool(
        name = "Python code interpreter",
        func=python_agent().run,
        description="useful for when you need to answer by running python code in a REPL."
    ),
    Tool(
        name = "CSV Agent",
        func=csv_agent().run,
        description="useful for when you need to answer questions about the recommended training plan."
    ),
]

prefix = """You are a kind AI agent acting as a cybersecurity advisor that belongs to OutThink, you are currently talking to a human \n
    Using the contexts below, answer the query. """
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

a = agent_chain.run(input="describe me the training plan proposed in a few words")

print(a)