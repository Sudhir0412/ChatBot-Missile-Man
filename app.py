import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import base64

## Load the API Keys from .env
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

st.title("Missile Man")
# -----------Display Dr. APJ Abdul Kalam's image as bakground-----------
# Function to encode the image file 
def get_base64(file_path): 
    with open(file_path, "rb") as f: 
        return base64.b64encode(f.read()).decode() 
# Custom CSS to set the background image with blur effect 
img_data = get_base64("./Dr_Kalam.png") 
page_bg_img = f''' <style> .stApp {{ background-image: url("data:image/png;base64,{img_data}"); 
background-size: cover; 
background-repeat: no-repeat; 
background-attachment: fixed; 
height: 100%; 
width: 100%; 
position: absolute; 
top: 0; left: 0; filter: blur(50%); }} 
</style> '''

st.markdown(page_bg_img, unsafe_allow_html=True)
#-----------------------------

llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide most accurate answer.
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./pdf") #Data ingestion
        st.session_state.docs=st.session_state.loader.load() #Data loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        
prompt1=st.text_input("Enter your question regarding Dr.APJ Kalam sir")

if st.button("Click me"):
    vector_embedding()
    st.markdown('<p style="color:#D14A8C; font-weight:bold;">Great question!!</p>', unsafe_allow_html=True)


    if prompt1:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        response=retrieval_chain.invoke({'input':prompt1})  
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            # Find relevant chunks
            for i,doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---------------------")
