import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
import os

load_dotenv()

st.set_page_config(page_title="PDF🙏")
# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] PDF RAG")

#캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")  

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("API Spec💬")

#처음 1번만 실행 하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = [] # 대화기록을 저장

if "chain" not in st.session_state:
    st.session_state["chain"] = None

#사이드바 생성
with st.sidebar:
     #초기화 버튼 생성
    clear_btn = st.button("대화내용 초기화")
    uploaded_files = st.file_uploader("파일 업로드",  accept_multiple_files=True, type=["PDF"])
    selected_model= st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
        )
   
    # 이전 대화 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)

# 새로운 메세지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

#파일 캐시 저장 (시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...") #cache 리소스는 오래걸리는 처리를 스피너를 사용해서 지루하지 않게? 또는 처리 중이라는 것을 보여주기 위해
def embed_files(files):
    all_split_documents = []
    
    for file in files:
        # 업로드한 파일을 캐시 디렉토리에 저장합니다.
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # 단계 1: 문서 로드(Load Documents)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # 단계 2: 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)
        
        all_split_documents.extend(split_documents)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=all_split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    
    return retriever


# 체인을 생성합니다.
def create_chain(retriever, model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding= "utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    return chain

if uploaded_files:
    #파일 업로드 후 retriever 생성(오래걸릴 예정)
    retriever = embed_files(uploaded_files)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

if clear_btn:
    st.session_state["messages"] = []

#이전 대화 기록 출력
print_messages()

#사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

#경고 메시지를 띄우기 위한 빈영역
warning_message = st.empty()


#사용자 입력이 들어오며
if user_input :
   
    #chain을 생성
    chain = st.session_state["chain"]

    if chain is not None:
         #사용자 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        #대화기록을 저장한다.        
        add_message("user", user_input)
        add_message("ai", ai_answer)
    else :
        warning_message.error("파일을 업로드 해주세요")