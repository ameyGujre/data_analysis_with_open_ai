import streamlit as st
from main_agent import DataAgent, load_data
from dataclasses import dataclass
from typing import Literal

# Dataclass for human/ai
@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["user", "assistant"]
    message: str


##Creating chat history holder in sesison state
if "history" not in st.session_state:
    st.session_state['history'] = []
    

##Designing side bar for loading file
with st.sidebar:
    st.title("Your personal Data Analyst")
    
    st.header("Upload the data below")

    file = st.file_uploader("Data Loader", type=['csv','txt'], label_visibility="hidden")
    if file != None:
        df = load_data(file)
        st.markdown("Click on save to initialize")
    
    if st.button("Save"):
        try:
            st.session_state['data_agent'] = DataAgent(df)
        except Exception as E:
            print(E)
            st.error("Upload the file first")


##Displaying and loading chat

for chat in st.session_state.history:
    with st.chat_message(chat.origin):
        st.markdown(chat.message, unsafe_allow_html=True)



if 'data_agent' in st.session_state:
    user_input = st.chat_input("Ask your query")
    if user_input:
        ##Displaying user input in chatbox
        with st.chat_message("user"):
            st.markdown(user_input, unsafe_allow_html=True)

        ##Adding user input into chat history
        st.session_state['history'].append(Message("user",user_input))

        ##Calling generate response function
        with st.spinner("Processing your query"):
            response = st.session_state['data_agent'].generate_response(user_input)
        ##Displaying the response into the chat box
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
        
        ##Adding the AI response into the chat history
        st.session_state['history'].append(Message("assistant",response))