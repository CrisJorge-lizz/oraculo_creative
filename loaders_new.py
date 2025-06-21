import os
from time import sleep
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import (WebBaseLoader,
                                                  YoutubeLoader,
                                                  CSVLoader,
                                                  PyPDFLoader,
                                                  TextLoader)
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

def carrega_lista(caminho):

    # print(f'Path: {caminho}')
    # print(f'Lista arquivos: {lista_PDFs}')

    # Carrega o CSV com os caminhos dos docs

    if not os.path.exists(caminho):
        st.error(f"O arquivo {caminho} não foi encontrado.")
        return ''

    df = pd.read_csv(caminho, on_bad_lines='warn', sep=';', engine='python')

    # Tratamento de erros iniciais

    if 'link' not in df.columns or 'type' not in df.columns:
        st.error("A coluna 'link' ou 'type' não foi encontrada no CSV.")
        return ''
    if df.empty:
        st.error("O CSV está vazio.")
        return ''

    missing_files = df[~df['link'].apply(os.path.exists)]['link'].tolist()
    if missing_files:
        st.error(f"Os arquivos não foram encontrados: {missing_files}")
        return ''
    if df.shape[0] > 10:
        st.warning("O CSV contém mais de 10 arquivos. Carregando apenas os primeiros 10.")
        df = df.head(10)

    print(f'Carregando {df.shape[0]} documentos da lista...')

    lista_documentos = []

    for _, row in df.iterrows():
        doc_type = str(row['type']).strip().upper()
        doc_path = row['link']

        if doc_type == 'PDF':
            loader = PyPDFLoader(doc_path)
        elif doc_type == 'TXT':
            loader = TextLoader(doc_path)
        elif doc_type == 'CSV':
            loader = CSVLoader(doc_path)
        elif doc_type == 'SITE':
            documento = carrega_site(doc_path)
            lista_documentos.append(type('Doc', (), {'page_content': documento})())
            continue
        elif doc_type == 'YOUTUBE':
            documento = carrega_youtube(doc_path)
            lista_documentos.append(type('Doc', (), {'page_content': documento})())
            continue
        else:
            st.warning(f"Tipo de arquivo '{doc_type}' não suportado para o arquivo {doc_path}.")
            continue

        documentos = loader.load()
        lista_documentos.extend(documentos)

    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento
