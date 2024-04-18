from io import BytesIO
import streamlit as st
import logging
import os
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
logo_url = 'https://govic.cl/ind/logo_web.png'

logging.getLogger('streamlit').setLevel(logging.ERROR)
st.set_page_config('Asistente Departamento Industrias Curso innovación')

col1, col2, col3 = st.columns([1,2,1])

with col2: # Usando la columna del medio para el logotipo
    st.image(logo_url, caption='', width=300) # Ajusta el ancho según necesites
st.header("Asistente Departamento Industria USACH")

#
# Aquí pones tu clave API de OpenAI directamente.
#OPENAI_API_KEY = ''

pdf_url = 'https://govic.cl/ind/innovacionEmprendimiento.pdf'

@st.cache_resource
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        return None

@st.cache_resource
def create_embeddings(pdf_bytes):
    pdf_reader = PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ''

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

# Descarga el PDF solo si se accede a la aplicación por primera vez o si la URL del PDF cambia.
pdf_content = download_pdf(pdf_url)

if pdf_content:
    knowledge_base = create_embeddings(pdf_content)
    user_question = st.text_input("Pregunta al asistente de Investigación de operaciones")

    if user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        docs = knowledge_base.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)

        st.write(respuesta)
else:
    st.error("No se pudo cargar el PDF desde la URL proporcionada.")
