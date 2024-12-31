import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROP_API_KEY")
model_name = os.getenv("MODEL_NAME")

from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0.5,groq_api_key=groq_api_key,
               model_name=model_name)

import pandas as pd
import uuid
import chromadb

chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection(name="my_collection")

def create_vector_db():
    data = pd.read_csv("fmb_faq.csv")

    for _,row in data.iterrows():
        collection.add(
            documents=[row['scrapped_text']],
            ids=[str(uuid.uuid4())]
        )

from langchain_core.prompts import PromptTemplate

def get_qa_chain(prompt_question):
    results = collection.query(
                    query_texts=[prompt_question], # Chroma will embed this for you
                    n_results=2 # how many results to return
              )
    page_data = results.get('documents')[0][0]

    prompt_extract = PromptTemplate.from_template(
    """
    ### TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION: REMOVE INTRODUCTION AND BULLET ITENS
    {prompt_question}
    ### (NO PREAMBLE)
    """
    )

    chain_extract = prompt_extract | llm

    res = chain_extract.invoke(input={'page_data': page_data, 'prompt_question': prompt_question})
    
    return res

create_vector_db()

import streamlit as st

st.title("Help Chat")
prompt_question = st.text_input("Enter the question :")
submit_button = st.button("Submit")

if submit_button:

    res = qa_chain(prompt_question)
    
    st.code(res.content, wrap_lines=True)
