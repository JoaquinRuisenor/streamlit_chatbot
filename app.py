import os
import pandas as pd
import tiktoken

from dotenv import load_dotenv

from langchain.vectorstores import Pinecone
import pinecone
import cohere
import streamlit as sl

from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


import openai
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from agent import rag_context, queryVector_db

load_dotenv()
pinecone.init(api_key="977b78cf-6233-481d-bb18-b88ec09fb444", environment="gcp-starter")
co = cohere.Client(os.environ["COHERE_API_KEY"])


# SETUP TPO QUERY ALREADY BUILT PINECONE VECTOR STORE DB

index_name = 'content-training-index'
embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))

# docsearch = Pinecone.from_documents(documents, embeddings, index_name = index_name)
docsearch = Pinecone.from_existing_index(embedding=embeddings,index_name=index_name)

index = pinecone.Index(index_name)
embed_model = "text-embedding-ada-002"



# SETUP THE LLM
template = """You are a kind AI agent that belongs to the Human Risk Management platform from OutThink and you were made with the intent to help cybersecurity admins plan their training effectively. you are currently talking to a human \n 
    Using the contexts below, answer the query.

    {context}

    givent he following:


    {chat_history}
    Human: {human_input}
    assistant:
    """

prompt = PromptTemplate(
    input_variables=["chat_history",  "human_input", "context"],
    template = template
)

# llm = ChatOpenAI(
#     openai_api_key = os.environ["OPENAI_API_KEY"],
#     model = 'gpt-3.5-turbo'
# )
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_chain(
    OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt
)






# SETUP THE APP
sl.set_page_config(
    page_title='OutThink Training Advisor',
    page_icon="",
    layout="wide"
)

sl.title('🤖 OutThink AI Advisor')

# Initialize session state
if "messages" not in sl.session_state.keys():
    sl.session_state.messages = [{"role": "assistant", "content": "Hello there, please upload your Cybersecurity Policy documents..."}]

# Display existing messages
for message in sl.session_state.messages:
    with sl.chat_message(message["role"]):
        sl.write(message["content"])

# Check if the file has been uploaded
if "file_uploaded_flag" not in sl.session_state:
    sl.session_state.file_uploaded_flag = False

# Sidebar for file uploader
uploaded_file = sl.sidebar.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None and not sl.session_state.file_uploaded_flag:
    thank_you_message = {"role": "assistant", "content": "Thank you for uploading the PDF! Ask me anything about cybersecurity to aid in your planning."}
    sl.write(thank_you_message["content"])
    sl.session_state.file_uploaded_flag = True

# Get user input and display chat
user_prompt = sl.chat_input()

if user_prompt is not None:
    sl.session_state.messages.append(
        {"role": "user", "content": user_prompt})
    with sl.chat_message("user"):
        sl.write(user_prompt)

if sl.session_state.messages[-1]["role"] != "assistant":
    with sl.chat_message("assistant"):
        with sl.spinner("Loading..."):
            context = queryVector_db(user_prompt, 3)
            ai_response = chain({"input_documents": context, "human_input": user_prompt}, return_only_outputs=True)
            ai_response = str(ai_response["output_text"])
            # ai_response = llm_chain.predict(query=user_prompt, creativity="high")
            sl.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    sl.session_state.messages.append(new_ai_message)

