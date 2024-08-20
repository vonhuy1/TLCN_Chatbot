import os

import streamlit as st
from bardapi import Bard

os.environ['_BARD_API_KEY'] = "dQg50xvxqQlaPFv-HIPCMdHRatpM2rvBJsUtI6qK-x1MCrG20brE4QEXp9e61BEogRimVw."

# Function to get Bard's response
def get_bard_response(question):
    # Create a Bard instance
    bard = Bard()
    # Get the answer using the Bard API
    return bard.get_answer(question)['content']

# Define a function to get and update chat history
def update_chat_history(user_question, chat_history):
    # Add user's question to chat history
    chat_history.append(f"You: {user_question}")
    
    # Get Bard's response and add it to chat history
    bard_response = get_bard_response(user_question)
    chat_history.append(f"Bot: {bard_response}")
    
    return chat_history

# Get the session state
session_state = st.session_state

# Initialize chat history if it doesn't exist in the session state
if 'chat_history' not in session_state:
    session_state.chat_history = []

st.title("Bard API Chat")

# Text input to get user's question
user_question = st.text_input("Enter your question", "")

# Button to get Bard's response
if st.button("Get Answer") and user_question:
    # Update chat history
    session_state.chat_history = update_chat_history(user_question, session_state.chat_history)

# Display chat history
chat_history = '\n'.join(session_state.chat_history)
st.markdown(f"<style>.reportview-container .main .block-container{{width: 1237px}}</style>", unsafe_allow_html=True)
st.text_area("Chat History", value=chat_history, height=485)
