import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import pandas as pd
import random
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Hugging Face chat model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=512
)
model = ChatHuggingFace(llm=llm)

# Generate synthetic user profiles
def generate_test_users(num_users=5):
    users = []
    # first_names = ["Alex", "Jamie", "Taylor", "Morgan", "Casey", "Riley"]
    first_names=["Zulu", "Waju", "Ali", "Umar", "Fatima", "Zainab"]
    issues = ["anxiety", "depression", "stress", "relationship problems", "work pressure"]
    coping_methods = ["breathing exercises", "journaling", "exercise", "meditation", "talking to friends"]
    
    for i in range(num_users):
        user = {
            "user_id": i+1,
            "name": random.choice(first_names),
            "age": random.randint(18, 45),
            "primary_issue": random.choice(issues),
            "preferred_coping": random.choice(coping_methods),
            "last_session": datetime.now().strftime("%Y-%m-%d")
        }
        users.append(user)
    
    return pd.DataFrame(users)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "test_users" not in st.session_state:
    st.session_state.test_users = generate_test_users()
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# System prompt
SYSTEM_PROMPT = """You are MIND, a mental health support chatbot..."""  # Keep your existing prompt

# Streamlit UI
st.title("🧠 MIND - Mental Health Support Chatbot")
st.caption("An AI-powered emotional support companion")

# User selection sidebar
with st.sidebar:
    st.header("Test User Profiles")
    user_option = st.selectbox(
        "Select a test user:",
        ["Select a user"] + list(st.session_state.test_users["name"])
    )
    
    if user_option != "Select a user":
        selected_user = st.session_state.test_users[st.session_state.test_users["name"] == user_option].iloc[0]
        st.session_state.current_user = selected_user.to_dict()  # Convert to dictionary
        st.write(f"**Age:** {st.session_state.current_user['age']}")
        st.write(f"**Primary Issue:** {st.session_state.current_user['primary_issue']}")
        st.write(f"**Preferred Coping:** {st.session_state.current_user['preferred_coping']}")
        st.write(f"**Last Session:** {st.session_state.current_user['last_session']}")

# Display conversation
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("How are you feeling today?"):
    if not st.session_state.current_user:
        st.warning("Please select a test user from the sidebar first.")
        st.stop()
    
    # Add user message
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    context = f"""
    [User Profile]
    Name: {st.session_state.current_user['name']}
    Age: {st.session_state.current_user['age']}
    Primary Issue: {st.session_state.current_user['primary_issue']}
    Preferred Coping: {st.session_state.current_user['preferred_coping']}
    """
    
    full_prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\nUser: {prompt}\n\nMIND:"
    
    with st.chat_message("assistant"):
        with st.spinner("MIND is thinking..."):
            try:
                result = model.invoke(full_prompt)
                response = result.content.split("MIND:")[-1].strip()
                st.session_state.conversation.append({"role": "assistant", "content": response})
                st.markdown(response)
            except Exception as e:
                st.error(f"Sorry, I encountered an error: {str(e)}")
                st.session_state.conversation.append({
                    "role": "assistant", 
                    "content": "I'm having trouble responding. Please try again later."
                })

# Resources section
st.sidebar.markdown("---")
st.sidebar.header("Quick Resources")
if st.session_state.current_user:
    st.sidebar.write(f"Try this for {st.session_state.current_user['primary_issue']}:")
    st.sidebar.markdown(f"- {st.session_state.current_user['preferred_coping']}")
st.sidebar.markdown("""
- [Breathing Exercise Guide](#)
- [Mental Health Hotlines](#)
- [Find a Therapist](#)
""")