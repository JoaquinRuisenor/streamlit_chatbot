import os
from dotenv import load_dotenv


import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent

# for the json loader
import json
from pathlib import Path
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chat_models import ChatOpenAI
from langchain.tools.json.tool import JsonSpec

# for pythopn agent
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent


from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone

import pinecone



load_dotenv()

def cybersecurity_policy_pdf_parser(path: str) -> RetrievalQA:
    """Retrieves data from a cybersecurity policy and compliance document for conversational context"""
    pdf_path = path
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    # Chunking
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separator="\n")
    docs = text_splitter.split_documents(documents = documents)

    # Embedding chunks into FAISS local vectorstore - Pinecone is on free trial
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")
    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)



    # This will send the query as a vector to our vector store and find ismilar vectors and esend it back with more conttext
    qa = RetrievalQA.from_chain_type(
        llm = OpenAI(),
        chain_type = "stuff",
        retriever = new_vectorstore.as_retriever())

    return qa


# def content_parser(path: str) -> RetrievalQA:
    
    
#     pinecone.init(api_key="977b78cf-6233-481d-bb18-b88ec09fb444", environment="gcp-starter")
#     embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
#     vectordb = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index)
#     retriever = vectordb.as_retriever()


#     text_field = "text"

#     # switch back to normal index for langchain
#     index = pinecone.Index(index_name)
#     vectorstore = Pinecone(
#         index, embed.embed_query, text_field
# )

#     # Initialize the OpenAI module, load and run the Retrieval Q&A chain
#     llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
#     qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
#     response = qa.run(query)

#     return qa







def python_agent():
    """Uses python"""
    python_agent_executor = create_python_agent(
        llm = ChatOpenAI(temperature=0, model='gpt-4'),
        tool = PythonREPLTool(),
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True
    )
    return python_agent_executor

def csv_agent():
    """loads csv file and creates agent to parse it"""
    path = 'C:\\Users\\JoaquinRuiseñor\\Documents\\OutThink DataScience\\LLM_Github\\chatbot-langchain\\training.csv'
    csv_agent = create_csv_agent(
        llm = ChatOpenAI(temperature=0, model='gpt-4'),
        path= path,
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True
    )
    return csv_agent



# file_path='C:\\Users\\JoaquinRuiseñor\\Documents\\OutThink DataScience\\LLM_Github\\chatbot-langchain\\training.json'
# data = json.loads(Path(file_path).read_text())
# # Usage
# loader = JSONLoader(file_path=file_path)
# print(loader.run("hello, how many campaigns"))
# 'q'
# pdf_path_input = 'C:\\Users\\JoaquinRuiseñor\\Documents\\OutThink DataScience\\LLM_Github\\CybersecurityDocs\\OutThink - Information Security Policy.pdf'
# json_path_input = 'C:\\Users\\JoaquinRuiseñor\\Documents\\OutThink DataScience\\LLM_Github\\chatbot-langchain\\training.json'

# docs = JSONLoader(path=json_path_input)
csv_path_input = 'C:\\Users\\JoaquinRuiseñor\\Documents\\OutThink DataScience\\LLM_Github\\chatbot-langchain\\training.csv'

csv_agent(csv_path_input).run('Name the first moduleName of the training plan')
'a'