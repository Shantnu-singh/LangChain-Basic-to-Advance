import streamlit as st
from LangChain_L11_Chatbot import chatbot , HumanMessage
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI

# st.set_page_config(layout="wide")

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['msg_hist'] = []
    
def load_conversation(thread_id):
    CONFIG = {'configurable': {'thread_id': thread_id}}

    return chatbot.get_state(config= CONFIG).values['messages']
    

# Create a session state
message_history = []

if 'msg_hist' not in st.session_state:
    st.session_state['msg_hist'] = []
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    
if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = []
    
add_thread(st.session_state['thread_id'])

for msg in st.session_state['msg_hist']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        
# SideBar UI
st.sidebar.title("SpiceJet Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()


st.sidebar.title("My Conversation")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        
        temp_msg = []
        
        for msg in messages:
            if isinstance(msg , HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_msg.append({'role':role , 'content' : msg.content})
        
        st.session_state['msg_hist'] = temp_msg
            
    
user_input = st.chat_input("Type here...")

if user_input:
    # First add user msg to hist
    st.session_state['msg_hist'].append({'role' : 'user' , 'content' : user_input })
    with st.chat_message("user"):
        st.markdown(user_input)

        CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

        # Save assistance msg in hist
        with st.chat_message("assistant"):
            ai_msg = st.write_stream(msg_chunk.content for msg_chunk,metadata in 
                            chatbot.stream({"messages" : [HumanMessage(content=user_input)]} , 
                                            config= CONFIG , 
                                            stream_mode='messages')
                            )
            st.session_state['msg_hist'].append({'role' : 'assistant' , 'content' : ai_msg})
