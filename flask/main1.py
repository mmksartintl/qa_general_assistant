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
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide de answer in Portuguese.

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

import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect

app = Flask(__name__)

#@app.route('/')
#def hello():
    #return 'Hello, World!'

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()
    return render_template('index.html', posts=posts)


@app.route('/create', methods=('GET', 'POST'))
def create():
    # if the user clicked on Submit, it sends post request
    if request.method == 'POST':
        # Get the title and save it in a variable
        title = request.form['title']
        # Get the content the user wrote and save it in a variable
        # content = request.form['content']
        if not title:
            flash('Title is required!')
        else:
            res = get_qa_chain(title)
            content = res.content
            # Open a connection to databse
            conn = get_db_connection()
            # Insert the new values in the db
            conn.execute('INSERT INTO posts (title, content) VALUES (?, ?)',(title, content))
            conn.commit()
            conn.close()
            # Redirect the user to index page
            return redirect(url_for('index'))

    return render_template('create.html')

