import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated 
import warnings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser 
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , START , END
from langgraph.checkpoint.memory import MemorySaver , InMemorySaver

warnings.filterwarnings("ignore")

# Load Models
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Parser
parser = StrOutputParser()

# call Models
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash" , api_key= GOOGLE_API_KEY)
# print(llm_gemini.invoke("who is father of india").content)

# Chatstate (to store all model)
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]
    
def chat_node(state :ChatState):
    
    # Take msg
    msg = state['messages']
    
    # send to llm
    responce = llm_gemini.invoke(msg)
    
    # responce 
    return {"messages" : responce}

# checkpointer
checkpointer = InMemorySaver()

# Making graph
graph = StateGraph(ChatState)

graph.add_node("Chat_node" , chat_node)
graph.add_edge(START , "Chat_node")
graph.add_edge("Chat_node" , END)

chatbot = graph.compile(checkpointer=checkpointer)

intial_state = {
    "messages" : [HumanMessage(content="what is the name of the indian Pm")]
}   

# print(chatbot.invoke({"messages" : [HumanMessage(content="Who is Indian PM?")]}))

thread_id = "1"

if __name__ == "__main__":
    while True:
        user_input = input("Type any message here.....:")
        
        if user_input.strip().lower() in ['exit' , 'no']:
            break
        
        config = {'configurable':{'thread_id':thread_id}}
        responce = chatbot.invoke({"messages" : [HumanMessage(content=user_input)]} , config= config)
        print("\nYour Msg: " , user_input)
        print("\nAI : " , responce['messages'][-1].content)
    



