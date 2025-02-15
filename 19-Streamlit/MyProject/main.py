import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

load_dotenv()

st.set_page_config(page_title="나만의 ChatGPT 💬", page_icon="💬")
st.title("HyAI💬")

#처음 1번만 실행 하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = [] # 대화기록을 저장



#사이드바 생성
with st.sidebar:
     #초기화 버튼 생성
    clear_btn = st.button("대화내용 초기화")

    import glob
    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox("프롬프트를 선택해 주세요", prompt_files, index=0)
    task_input = st.text_input("Task 입력", "")

    # if user_text_apply_btn:
    #     tab1.markdown(f"✅ 프롬프트가 적용되었습니다")
    #     prompt_template = user_text_prompt + "\n\n#Question:\n{question}\n\n#Answer:"
    #     prompt = PromptTemplate.from_template(prompt_template)
    #     st.session_state["chain"] = create_chain(prompt, "gpt-4o")

    # user_selected_apply_btn = tab2.button("프롬프트 적용", key="apply2")
    # if user_selected_apply_btn:
    #     tab2.markdown(f"✅ 프롬프트가 적용되었습니다")
    #     prompt = load_prompt(f"{selected_prompt}", encoding="utf8")
    #     st.session_state["chain"] = create_chain(prompt, "gpt-4o-mini")


# 이전 대화 출력
def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)

# 새로운 메세지 추가
def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 체인을 생성합니다.
def create_chain(prompt_filepath, task=""):
    #prompt 적용
    prompt = load_prompt(prompt_filepath, encoding= "utf-8")
    if task :
        prompt = prompt.partial(task=task)
    
    
   # 체인생성 =  prompt | GPT  | 출력 파서
    chain = prompt | ChatOpenAI(model_name="gpt-4o", temperature=0) | StrOutputParser()
    return chain

if clear_btn:
    retriever = st.session_state["messages"].clear()

# if "chain" not in st.session_state:
#     # user_prompt
#     prompt_template = selected_prompt + "\n\n#Question:\n{question}\n\n#Answer:"
#     prompt = PromptTemplate.from_template(prompt_template)
#     st.session_state["chain"] = create_chain(prompt, "gpt-4o-mini")

#사용자 입력
print_history()

if user_input := st.chat_input("궁금한 내용을 물어보세요!"):
    #사용자 입력
    st.chat_message("user").write(user_input)
    #chain을 생성
    chain = create_chain(selected_prompt, task = task_input)
    add_history("user", user_input)
    
    # 스트리밍 호출
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        chat_container = st.empty()

        ai_answer = ""
        for chunk in response:
            ai_answer += chunk
            chat_container.markdown(ai_answer)

    #대화기록을 저장한다.        
    add_history("user", user_input)
    add_history("ai", ai_answer)
