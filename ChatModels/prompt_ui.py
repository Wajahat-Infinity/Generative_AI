import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Hugging Face chat model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=10
)
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("🤖 AI Chatbot with Hugging Face")

# User input
prompt = st.text_input("Enter your prompt:", "")

# On button click
if st.button("Send"):
    if prompt.strip():
        with st.spinner("Thinking..."):
            result = model.invoke(prompt)
            st.success("Response:")
            st.write(result.content)
    else:
        st.warning("Please enter a prompt.")


#---------------------------------------------------------------------------
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# import streamlit as st

# # Load environment variables
# load_dotenv()

# # Initialize the chat model
# @st.cache_resource
# def load_chat_model():
#     llm = HuggingFaceEndpoint(
#         repo_id="HuggingFaceH4/zephyr-7b-beta",
#         task="text-generation",
#         temperature=0.1,
#         max_new_tokens=512
#     )
#     return ChatHuggingFace(llm=llm)

# model = load_chat_model()

# # Streamlit app
# st.title("🤖 AI Chatbot with Hugging Face")
# st.write("Ask me anything about AI and its impact on the future!")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("What would you like to know?"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         response = model.invoke(prompt)
#         st.markdown(response.content)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response.content})