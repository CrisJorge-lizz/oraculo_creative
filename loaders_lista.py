import os
from time import sleep
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import (WebBaseLoader,
                                                  YoutubeLoader,
                                                  CSVLoader,
                                                  PyPDFLoader,
                                                  TextLoader)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from fake_useragent import UserAgent

def carrega_site(url):
    documento = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
            break
        except:
            print(f'Erro ao carregar o site {i+1}')
            sleep(3)
    if documento == '':
        st.error('Não foi possível carregar o site')
        st.stop()
    return documento

def carrega_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info=False, language=['pt'])
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_txt(caminho):
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_lista_txt():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "txt_clean")
    if not os.path.exists(data_dir):
        st.error(f"Pasta {data_dir} não encontrada.")
        return []
    arquivos_txt = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not arquivos_txt:
        st.warning("Nenhum arquivo .txt encontrado na pasta data/txt_clean.")
        return []
    documentos = []
    for caminho in arquivos_txt:
        try:
            loader = TextLoader(caminho)
            docs = loader.load()
            # Garante que cada doc tenha metadata['source'] preenchido
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = caminho
            documentos.extend(docs)
        except Exception as e:
            st.warning(f"Erro ao ler {caminho}: {e}")
    st.info(f'Carregando {len(arquivos_txt)} arquivos .txt da pasta data/txt_clean...')
    return documentos
