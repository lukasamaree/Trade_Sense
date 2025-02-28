import streamlit as st
import pandas as pd
import requests
import os
# import finnhub
import pandas
from datetime import datetime, timedelta
import re
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
import time
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from play import *





# Retrieve original Finhubb API
path = '/Users/lukasamare/Desktop/fall_module_1/communications/'
if os.path.exists(path):
    with open(path+'final_project/finhubb_api.txt','r') as file:
        lukas_fin_api = file.read().strip()
else:
    lukas_fin_api = ''

# Retrieve ChatGPT API Key
path1 = "/Users/lukasamare/Desktop/random_project"
if os.path.exists(path1):
    with open(path1+"openaiapikey", "r") as file:
        apikey = file.read().strip()
else:
    apikey = st.secrets["OPENAI_API_KEY"]

# Make Title for the streamlit app
st.title("Due Diligance App")
if apikey:
    st.write("hello")

# Sign in to input finhubb api
# st.write("#### ")
# st.markdown("Check out this [website](%s) to get your free finhubb api code, if the original code doesn't work. " % "https://finnhub.io/register")
# finhubb_api  = st.text_input(
#     label="Enter your Finhubb API code",
#     value = lukas_fin_api,
#     help  = " Use this link https://finnhub.io/register \n to get free finhubb api code if original one doesn't work")

st.write("#### Do you have an OPEN AI API KEY?")

if "show_text_input" not in st.session_state:
    st.session_state.show_text_input = False
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

# Buttons for Yes and No
col1, col2 = st.columns(2)
with col1:
    if st.button("Yes"):
        st.session_state.show_text_input = True
        st.session_state.selected_option = None  # Reset variable if switching

with col2:
    if st.button("No"):
        st.session_state.show_text_input = False
        st.session_state.selected_option = "This bum does not have an API key, GET YOUR BREAD UP AND PAY. jk I have one for you"

# Show text input if Yes is clicked
if st.session_state.show_text_input:
    st.markdown("Check out this [website](%s) to get your Open AI API code, if the original code doesn't work. " % "https://platform.openai.com/api-keys")
    apikey  = st.text_input(
    label="Enter your Open AI API",
    placeholder="Type your Open AI API",
    help  = " Use this link https://platform.openai.com/api-keys \n to get Open API key if original one doesn't work"
    )   

# Show selected option if No is clicked
if st.session_state.selected_option == "This bum does not have an API key, GET YOUR BREAD UP AND PAY. jk I have one for you":
    st.write(st.session_state.selected_option)
    apikey = st.secrets["OPENAI_API_KEY"]

# Retrieve Open AI API


#Make prompt for specific ticker you want to look up
st.write("#### What stock do you want your DD on?")

# Stock ticker Answer
stock_ticker = st.text_input(
    label="Enter your stock ticker:",
    value="",
    max_chars=50,
    placeholder="Type your stock ticker here",
    help="This is where you input your stock ticker"
)

# Retrieve Docs
urls = retrieve_urls_for_docs(stock_ticker)

# Webscrape Documents from the Web
def get_documents_from_web(urls):
    doc_lst = []
    for url in urls:
        loader =  WebBaseLoader(url)
        docs = loader.load()
        doc_lst.extend(docs)
        # print(doc_lst)
    splitter =  RecursiveCharacterTextSplitter(
        chunk_size = 200
    )
    split_docs =  splitter.split_documents(doc_lst)
    return split_docs

docs  = get_documents_from_web(urls)

# Create Vector Database
def create_db(docs):
    embedding =  OpenAIEmbeddings(api_key=apikey)
    vector_store = FAISS.from_documents(docs,embedding=embedding)
    return vector_store

vectorStore = create_db(docs)
    
# Create Chain
def create_chain(vectorStore):
    llm1 = ChatOpenAI(api_key=apikey,
    temperature = 0.2,
    model = "gpt-4-turbo",
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI financial advisor. Give me recent sentiment and updated news on stock, is it bullish or bearish or in between. What are the goals of the company? What seperates this company from other companies from this market?"),
        ("human", '{input}'),
        ("user", '{context}')

    ])
    
    chain = create_stuff_documents_chain(llm=llm1, prompt=prompt)

    retriever = vectorStore.as_retriever(search_kwargs = {'k':10})
    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain
    

chain = create_chain(vectorStore)

response =  chain.invoke({"input" : stock_ticker})

elapsed = time.time()

if st.button("Enter"):
    st.write("Summary: "+response["answer"])   

print(time.time() - elapsed)




