# app.py

# Importando todas as bibliotecas que instalamos
import streamlit as st
import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from zipfile import ZipFile
from langchain_community.llms import OpenAI # Import corrigido
from langchain_experimental.agents import create_csv_agent

# --- 1. Configuração da Página e Conexão com Firebase ---

# st.set_page_config deve ser o primeiro comando Streamlit a ser executado
st.set_page_config(page_title="Análise de CSV com IA", layout="wide")

# Função para conectar ao Firebase (usando cache para eficiência)
@st.cache_resource
def init_firebase_connection():
    """
    Usa as credenciais do st.secrets para inicializar a conexão com o Firebase
    e retorna o cliente do Firestore.
    """
    # LINHA MODIFICADA: Convertendo o objeto de segredos para um dicionário padrão.
    creds_dict = dict(st.secrets["firebase_credentials"])
    
    creds = credentials.Certificate(creds_dict)
    firebase_admin.initialize_app(creds)
    return firestore.client()

# Tenta inicializar a conexão. Se der erro, para o app.
try:
    if not firebase_admin._apps:
        db = init_firebase_connection()
    else:
        db = firestore.client()
except Exception as e:
    st.error(f"Falha ao conectar com o Firebase. Verifique seu `secrets.toml`. Erro: {e}")
    st.stop()

# --- 2. Interface do Usuário (UI) com Streamlit ---

st.title("🤖 Análise de CSV com Inteligência Artificial")
st.markdown("Faça upload de um arquivo `.zip` contendo um ou mais CSVs e faça perguntas em português!")

# Função para descompactar o arquivo e listar os CSVs
def process_zip_file(uploaded_file, extract_path="temp_csvs"):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    with ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    return [f for f in os.listdir(extract_path) if f.endswith('.csv')], extract_path

# --- 3. Lógica Principal do Aplicativo ---

# Cria duas colunas para organizar a interface
col1, col2 = st.columns(2)

with col1:
    uploaded_zip = st.file_uploader("1. Faça o upload do seu arquivo .ZIP aqui", type="zip")

# Se um arquivo foi enviado, o código abaixo é executado
if uploaded_zip:
    csv_files, folder_path = process_zip_file(uploaded_zip)

    if not csv_files:
        st.warning("Nenhum arquivo .csv encontrado no .zip enviado.")
    else:
        with col1:
            selected_csv = st.selectbox("2. Escolha o CSV para analisar:", options=csv_files)
            selected_csv_path = os.path.join(folder_path, selected_csv)

        with col2:
            question = st.text_area("3. Faça sua pergunta sobre o arquivo selecionado:")

            if st.button("Perguntar à IA"):
                if not question:
                    st.warning("Por favor, digite uma pergunta.")
                else:
                    with st.spinner("A IA está analisando os dados e pensando..."):
                        try:
                            # Configura o modelo da OpenAI lendo a chave do st.secrets
                            llm = OpenAI(
                                temperature=0,
                                openai_api_key=st.secrets["OPENAI_API_KEY"]
                            )

                            # Cria o agente da LangChain que sabe "conversar" com CSVs
                            agent = create_csv_agent(
                                llm,
                                selected_csv_path,
                                verbose=True, # Mostra o passo a passo do pensamento da IA no terminal
                                allow_dangerous_code=True,
                            )

                            # Executa a pergunta
                            answer = agent.run(question)

                            # Exibe a resposta
                            st.success("#### Resposta da IA:")
                            st.markdown(answer)

                            # Salva a interação no Firebase
                            log_ref = db.collection('interacoes').document()
                            log_ref.set({
                                'arquivo_csv': selected_csv,
                                'pergunta': question,
                                'resposta': answer,
                                'timestamp': firestore.SERVER_TIMESTAMP
                            })

                        except Exception as e:
                            st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
        
        st.divider()
        with st.expander("Clique para ver uma prévia dos dados do CSV selecionado"):
            df = pd.read_csv(selected_csv_path)
            st.dataframe(df.head())