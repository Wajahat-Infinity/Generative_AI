import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize all modules
@st.cache_resource
def initialize_system():
    # Module 2: Sentiment Analysis
    sentiment_analyzer = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis")
    
    # Module 3 & 4: LLM with RAG capabilities
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature=0.5,  # More creative responses
        max_new_tokens=512
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    # Sample self-help resources (Module 4)
    resources = [
        "Breathing exercise: Inhale for 4 seconds, hold for 7, exhale for 8.",
        "Mindfulness technique: Focus on 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
        "Stress management: Break tasks into smaller steps and prioritize them.",
        "Anxiety relief: Practice grounding techniques by naming objects around you."
    ]
    
    # Create vector store for resources (RAG implementation)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    resource_db = FAISS.from_texts(resources, embeddings)
    
    return {
        "sentiment_analyzer": sentiment_analyzer,
        "chat_model": chat_model,
        "resource_db": resource_db
    }

system = initialize_system()

# Custom prompt template for mental health support (Module 3)
MENTAL_HEALTH_PROMPT = ChatPromptTemplate.from_template(
    """You are MIND, a mental health support assistant. Your role is to provide empathetic, 
    non-judgmental support to users experiencing stress, anxiety, or depression.
    
    User's emotional state: {sentiment}
    Relevant self-help resource: {resource}
    
    Current conversation: {history}
    
    User: {input}
    MIND:"""
)

# Streamlit app interface (Module 1)
st.title("🧠 MIND: Mental Intelligence for Nurturing Dialogue")
st.caption("An AI-powered mental health support chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello, I'm MIND. How are you feeling today?"}]
    st.session_state.sentiment_history = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("Share your thoughts or feelings..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Module 2: Analyze sentiment
    sentiment_result = system["sentiment_analyzer"](prompt)[0]
    current_sentiment = sentiment_result["label"]
    st.session_state.sentiment_history.append(current_sentiment)
    
    # Module 4: Retrieve relevant self-help resource
    relevant_resource = system["resource_db"].similarity_search(prompt, k=1)[0].page_content
    
    # Prepare context for response generation
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-4:]])
    
    # Generate response (Modules 1 & 3)
    with st.chat_message("assistant"):
        with st.spinner("MIND is thinking..."):
            # Format the prompt with all context
            formatted_prompt = MENTAL_HEALTH_PROMPT.format(
                sentiment=current_sentiment,
                resource=relevant_resource,
                history=chat_history,
                input=prompt
            )
            
            # Get response from LLM
            response = system["chat_model"].invoke(formatted_prompt)
            
            # Display response
            st.markdown(response.content)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response.content})

# Module 5: Privacy notice
st.sidebar.markdown("### Privacy Notice")
st.sidebar.info("""
All conversations are anonymous and not stored permanently. 
We don't collect personal information and cannot replace professional help.
""")

# Module 6: Feedback option
st.sidebar.markdown("### Feedback")
feedback = st.sidebar.radio("Was this helpful?", ["", "Yes", "Somewhat", "No"], index=0)
if feedback:
    st.sidebar.success("Thank you for your feedback!")


# import streamlit as st
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from transformers import pipeline

# # Load environment variables
# load_dotenv()

# # Setup LLM and sentiment analysis
# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     temperature=0.1,
#     max_new_tokens=512
# )
# model = ChatHuggingFace(llm=llm)
# sentiment_pipeline = pipeline("sentiment-analysis")

# # Streamlit app layout
# st.set_page_config(page_title="MIND - Mental Health Chatbot", layout="centered")
# st.title("🧠 MIND: Mental Health Support Chatbot")
# st.markdown("A chatbot that listens, understands, and supports you.")

# # Chat history state
# if "chat" not in st.session_state:
#     st.session_state.chat = []

# # User input
# user_input = st.text_input("How are you feeling today?", key="user_input")

# if st.button("Send") and user_input:
#     # Step 1: Analyze Sentiment
#     sentiment = sentiment_pipeline(user_input)[0]
#     emotion = sentiment["label"]
#     confidence = round(sentiment["score"], 2)

#     # Step 2: Build Dynamic Prompt
#     base_instruction = {
#         "POSITIVE": "Respond in a friendly and encouraging way.",
#         "NEGATIVE": "Respond empathetically and offer helpful coping strategies.",
#         "NEUTRAL": "Respond calmly and neutrally, ask supportive follow-ups."
#     }
#     instruction = base_instruction.get(emotion.upper(), "Respond with care.")

#     dynamic_prompt = f"The user said: \"{user_input}\". They seem to be feeling {emotion.lower()} (confidence: {confidence}). {instruction}"

#     # Step 3: Get AI Response
#     response = model.invoke(dynamic_prompt).content

#     # Step 4: Update chat history
#     st.session_state.chat.append(("You", user_input))
#     st.session_state.chat.append(("MIND", response))

# # Display chat
# for sender, message in st.session_state.chat:
#     with st.chat_message(sender.lower()):
#         st.markdown(message)
