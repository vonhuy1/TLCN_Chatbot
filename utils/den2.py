import os
import streamlit as st
from bardapi import Bard
os.environ['_BARD_API_KEY'] = "cgjOKgkF5qUxhJNoy0HWGAPHON0SPO6rC60jDKfrkaLrmf7N8HT1fKBWJqywHiwfRd_qOQ."

def get_bard_response(question):
    bard = Bard()
    return bard.get_answer(question)['content']

def update_chat_history(user_question, chat_history):
    chat_history.append(f"You: {user_question}")
    bard_response = get_bard_response(user_question)
    chat_history.append(f"Bot: {bard_response}")
    return chat_history

session_state = st.session_state
if 'chat_history' not in session_state:
    session_state.chat_history = []
st.title("Bard API Chat")
user_question = st.text_input("Enter your question", "")

if st.button("Get Answer") and user_question:
    session_state.chat_history = update_chat_history(user_question, session_state.chat_history)

chat_history = '\n'.join(session_state.chat_history)
st.markdown(f"<style>.reportview-container .main .block-container{{width: 1237px}}</style>", unsafe_allow_html=True)
st.text_area("Chat History", value=chat_history, height=485)
