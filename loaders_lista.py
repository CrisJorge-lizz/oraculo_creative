import os
from time import sleep
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
from fake_useragent import UserAgent

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
    if not DATA_TXT_DIR.exists():
        st.error(f"Pasta {DATA_TXT_DIR} não encontrada.")
        return []

    arquivos = sorted(DATA_TXT_DIR.glob("*.txt"))
    if not arquivos:
        st.warning(f"Nenhum .txt em {DATA_TXT_DIR}.")
        return []

    documentos = []
    for caminho in arquivos:
        loader = TextLoader(str(caminho))
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", str(caminho.relative_to(ROOT_DIR)))
            doc.metadata["sha256"] = _sha256(caminho)
        documentos.extend(docs)

    st.info(f"Carregados {len(arquivos)} arquivos de {DATA_TXT_DIR}.")
    return documentos

# --- Construção/Persistência do índice ---------------------------------
def get_vectorstore(project: str, api_key: str | None = None):
    """
    Devolve (vectorstore, created_now:boolean).
    Salva/recupera em  …/indices/<project>/index.faiss
    """
    project_dir = INDEX_BASE_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    index_file = project_dir / "index.faiss"

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    if index_file.exists():
        vs = FAISS.load_local(str(project_dir), embeddings, allow_dangerous_deserialization=True)
        return vs, False

    docs = carrega_lista_txt()
    if not docs:
        st.stop()

    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(project_dir))
    return vs, True
