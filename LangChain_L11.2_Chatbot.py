import streamlit as st
from LangChain_L11_Chatbot import chatbot , HumanMessage
CONFIG = {'configurable': {'thread_id': 'thread-1'}}
# st.set_page_config(layout="wide")

# Create a session state
message_history = []

if 'msg_hist' not in st.session_state:
    st.session_state['msg_hist'] = []

for msg in st.session_state['msg_hist']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

    
user_input = st.chat_input("Type here...")

if user_input:
    # First add user msg to hist
    st.session_state['msg_hist'].append({'role' : 'user' , 'content' : user_input })
    with st.chat_message("user"):
        st.markdown(user_input)

        # Save assistance msg in hist
        with st.chat_message("assistant"):
            ai_msg = st.write_stream(msg_chunk.content for msg_chunk,metadata in 
                            chatbot.stream({"messages" : [HumanMessage(content=user_input)]} , 
                                            config= CONFIG , 
                                            stream_mode='messages')
                            )
            st.session_state['msg_hist'].append({'role' : 'assistant' , 'content' : ai_msg})
