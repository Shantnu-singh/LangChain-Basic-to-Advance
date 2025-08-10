import os
import google.generativeai as palm
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt
load_dotenv()

# Load your keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash" , api_key= GOOGLE_API_KEY)

st.title("Ai Research Assistance")

# user_input = st.text_input("Enter you promt here...")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt("Data\promt_template_reseach_paper.json")


if st.button("Summurise"):
    chain = template | llm_gemini
    results = chain.invoke({
        'paper_input' : paper_input,
        'length_input' : length_input,
        'style_input' : style_input
    })
    # Fill the placeholder
    # promt = template.invoke)
    # results = llm_gemini.invoke(promt)
    st.write(results.content)