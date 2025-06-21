import os
from time import sleep
import time
import streamlit as st

from pathlib import Path
import hashlib

from langchain_community.document_loaders import (WebBaseLoader,
                                                  YoutubeLoader,
                                                  CSVLoader,
                                                  PyPDFLoader,
                                                  TextLoader)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from fake_useragent import UserAgent
os.environ['USER_AGENT'] = UserAgent().random

ROOT_DIR       = Path(__file__).resolve().parent          #  …/seu-projeto
DATA_TXT_DIR   = ROOT_DIR / "data" / "txt_clean"          #  …/data/txt_clean
INDEX_BASE_DIR = ROOT_DIR / "indices"                     #  …/indices
INDEX_BASE_DIR.mkdir(exist_ok=True)

def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def carrega_site(url):
    documento = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(
                url,
                raise_for_status=True,
                requests_kwargs={"headers": {"User-Agent": os.environ['USER_AGENT']}}
            )
            lista_documentos = loader.load()
            documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
            break
        except Exception as e:
            print(f'Erro ao carregar o site {i+1}: {e}')
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
    st.info(f"Encontrados {len(arquivos_txt)} arquivos.")
    if not arquivos_txt:
        st.warning("Nenhum arquivo .txt encontrado na pasta data/txt_clean.")
        return []
    documentos = []
    for caminho in arquivos_txt:
        st.info(f"Lendo {caminho} ({os.path.getsize(caminho)} bytes)")
        t0 = time.time()
        try:
            loader = TextLoader(caminho)
            docs = loader.load()
            st.info(f"Leitura levou {time.time()-t0:.2f} segundos")
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = caminho
            documentos.extend(docs)
        except Exception as e:
            st.warning(f"Erro ao ler {caminho}: {e}")
    st.info(f'Carregando {len(arquivos_txt)} arquivos .txt da pasta data/txt_clean...')
    return documentos

def get_vectorstore(project: str, api_key: str | None = None, provedor: str = "Ollama", modelo: str = "llama3.2:3b"):
    """
    Devolve (vectorstore, created_now:boolean).
    Salva/recupera em  …/indices/<project>/index.faiss
    """
    project_dir = INDEX_BASE_DIR / f"{project}_{provedor}_{modelo.replace(':', '_')}"
    project_dir.mkdir(parents=True, exist_ok=True)
    index_file = project_dir / "index.faiss"

    if provedor == "Ollama":
        embeddings = OllamaEmbeddings(model=modelo)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    if index_file.exists():
        vs = FAISS.load_local(str(project_dir), embeddings, allow_dangerous_deserialization=True)
        return vs, False

    docs = carrega_lista_txt()
    if not docs:
        st.error("Nenhum documento encontrado.")
        st.stop()

    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(project_dir))
    return vs, True
