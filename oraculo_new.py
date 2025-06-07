import streamlit as st
from langchain.memory import ConversationBufferMemory
#                               avaliar outros tipos de memoria

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from langchain.prompts import ChatPromptTemplate

from loaders_new import *
import tempfile

import json

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'PDF', 'CSV', 'TXT', 'CSV com Lista de documentos',
]

CONFIG_MODELOS = {'OpenAI':
                        {'modelos': ['gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini'],
                         'chat': ChatOpenAI},
                  'Groq':
                        {'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
                         'chat': ChatGroq}}

MEMORIA = ConversationBufferMemory()

MEMORY_DICT = {"chat_history":[]}

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        return carrega_site(arquivo)
    elif tipo_arquivo == 'Youtube':
        return carrega_youtube(arquivo)
    elif tipo_arquivo == 'PDF':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        return carrega_pdf(nome_temp)
    elif tipo_arquivo == 'CSV':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        return carrega_csv(nome_temp)
    elif tipo_arquivo == 'TXT':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        return carrega_txt(nome_temp)
    elif tipo_arquivo == 'CSV com Lista de documentos':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        return carrega_lista(nome_temp)
    else:
        st.error("Tipo de arquivo não suportado.")
        return None


def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):

    documento = carrega_arquivos(tipo_arquivo, arquivo)

    system_message = '''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas
    de um documento {}:

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substitua por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue"
    sugira ao usuário carregar novamente o Oráculo!'''.format(tipo_arquivo, documento)

    print(system_message)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain

def pagina_chat():
    global MEMORY_DICT
    st.header("🤯 Sou o Oraculo da Creativity in Vitro 🧪", divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Inicialize o Oráculo 🤯')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Abra seu coração 🩶')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario,
            'chat_history': memoria.buffer_as_messages
            }))

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

        # Extrai o buffer como dicionário serializável
        MEMORY_DICT = {"chat_history": MEMORIA.load_memory_variables({})["history"]}


def sidebar():
    global MEMORY_DICT
    st.sidebar.title("Configurações do Oráculo")
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox("Selecione o tipo de arquivo", TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
        if tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
        if tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do arquivo csv', type=['.csv'])
        if tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo txt', type=['.txt'])
        if tipo_arquivo == 'Lista de PDFs':
            arquivo = st.file_uploader('Faça o upload do arquivo csv com a lista', type=['.csv'])
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelos', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a api key para o provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}'),
            type="password"
        )

        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar Oráculo', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

# Botão para salvar histórico como JSON
    if st.button("Salvar Histórico de Conversa", use_container_width=True):
        memoria = st.session_state.get('memoria', MEMORIA)
        historico = memoria.load_memory_variables({})["history"]
        json_data = json.dumps({"chat_history": historico}, indent=4, ensure_ascii=False)
        with open("conversa.json", "w") as f:
            f.write(json_data)
        st.success("Conversa salva com sucesso!")

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
