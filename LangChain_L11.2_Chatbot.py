import streamlit as st
from LangChain_L11_Chatbot import chatbot , HumanMessage
CONFIG = {'configurable': {'thread_id': 'thread-1'}}


# Create a session state
message_history = []

if 'msg_hist' not in st.session_state:
    st.session_state['msg_hist'] = []

for msg in st.session_state['msg_hist']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

# with st.chat_message("user"):
#     st.text("HI")
    
# with st.chat_message("assistant"):
#     st.text("Hey to you too..")
    
# with st.chat_message("user"):
#     st.text("My name is shantnu")

# with st.chat_message("assistant"):
#     st.text("Hey Shantnu, How can i help you.")
    
user_input = st.chat_input("Type here...")

if user_input:
    # First add user msg to hist
    st.session_state['msg_hist'].append({'role' : 'user' , 'content' : user_input })
    with st.chat_message("user"):
        st.text(user_input)
    
    responce = chatbot.invoke({"messages" : [HumanMessage(content=user_input)]} , config= CONFIG)
    # Save assistance msg in hist
    st.session_state['msg_hist'].append({'role' : 'assistant' , 'content' : responce['messages'][-1].content})
    with st.chat_message("assistant"):
        st.text(responce['messages'][-1].content)