import os
import pandas as pd
import tiktoken

from dotenv import load_dotenv

from langchain.vectorstores import Pinecone
import pinecone
import cohere

from langchain_community.chat_models import ChatOpenAI
import openai



from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings




load_dotenv()
pinecone.init(api_key="977b78cf-6233-481d-bb18-b88ec09fb444", environment="gcp-starter")
co = cohere.Client(os.environ["COHERE_API_KEY"])

chat = ChatOpenAI(
    openai_api_key = os.environ["OPENAI_API_KEY"],
    model = 'gpt-3.5-turbo'
)





# SETUP TPO QUERY ALREADY BUILT PINECONE VECTOR STORE DB

index_name = 'content-training-index'
embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))

# docsearch = Pinecone.from_documents(documents, embeddings, index_name = index_name)
docsearch = Pinecone.from_existing_index(embedding=embeddings,index_name=index_name)

index = pinecone.Index(index_name)
embed_model = "text-embedding-ada-002"


# DEFINE embedding function
def embed(docs: list[list[float]]):
    res = openai.Embedding.create(input=docs, engine = embed_model)
    embeddings = [x["embedding"] for x in res["data"]]
    return embeddings
def get_docs(query: str, top_k: int, metadata_fields: list = None):
    # encode query
    xq = embed([query])
    
    # search pinecone index
    res = index.query(xq, top_k=top_k, include_metadata=True)
    
    # get doc text and metadata
    docs = [{"text": x["metadata"]["text"], "metadata": {field: x["metadata"][field] for field in metadata_fields}}
            if metadata_fields else {"text": x["metadata"]["text"]} 
            for x in res["matches"]]
    
    return docs


# docs = get_docs(query, top_k=20, metadata_fields=metadata_fields)


# initialize the vector store object


def rag_context(query: str):
    # get top 3 results from knowledge base
    openai.api_key = os.environ["OPENAI_API_KEY"]
    metadata_fields = ["moduleName", "moduleId"]
    docs = get_docs(query, top_k=3, metadata_fields=metadata_fields)
    print(docs)
    # get the text from the results
    source_knowledge = "\n".join([f"ModuleId: {x['metadata']['moduleId']}\n Module Content:{x['text']}" for x in docs])
    return str(source_knowledge)

def queryVector_db(query: str, k: int):
    docs = docsearch.similarity_search(query, k=k)
    return docs

query = 'Please show me training module names on Passwords'
final_documents = queryVector_db(query, 3)


# rerank_docs = co.rerank(
#     query=query, documents=docs, top_n=12, model = "rerank-english-v2.0"
# )
# print(rerank_docs)
# chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
'aaaa'