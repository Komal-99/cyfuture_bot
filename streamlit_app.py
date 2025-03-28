
# import streamlit as st
# import requests
# import time

# # Set up Streamlit UI
# st.set_page_config(page_title="AI Chatbot", layout="centered")
# st.title("🤖 AI Chatbot")

# # Store conversation history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Function to query the AI API
# def query_ai(question):
#     url = "http://127.0.0.1:7860/query/"
#     params = {"input_text": question}
#     response = requests.get(url, params=params, stream=True)

#     if response.status_code == 200:
#         full_response = ""
#         for chunk in response.iter_content(chunk_size=1024):
#             if chunk:
#                 text = chunk.decode("utf-8")
#                 full_response += text
#                 yield text  # Streaming raw response
#         # Extract only the refined response after </think>
#         refined_answer = full_response.split("</think>", 1)[-1].strip()
#         yield f"✅ **Refined Answer:**\n{refined_answer}"
#     else:
#         yield "❌ Error: Unable to fetch response."

# # Custom CSS to fix paragraph spacing and format response box
# st.markdown("""
#     <style>
#         .chat-box {
#             background-color: #1e1e1e;
#             padding: 15px;
#             border-radius: 10px;
#             margin-top: 10px;
#             font-size: 12px;
#             font-family: monospace;
#             white-space: pre-wrap;
#             word-wrap: break-word;
#             line-height: 1;  /* Fixes paragraph spacing */
#         }
#     </style>
# """, unsafe_allow_html=True)

# # User input area
# user_input = st.text_input("Ask a question:", "", key="user_input")
# submit_button = st.button("Submit")

# if submit_button and user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     st.session_state.messages.append({"role": "bot", "content": ""})  # Placeholder

#     response_text = ""

#     message_container = st.empty()
#     with message_container:
#         for chunk in query_ai(user_input):
#             response_text += chunk
#             st.session_state.messages[-1]["content"] = response_text  # Update last AI response
#             message_container.empty()  # Clear previous output
#             st.markdown(f'<div class="chat-box">{response_text}</div>', unsafe_allow_html=True)
#             time.sleep(0.1)  # Smooth streaming effect
import streamlit as st
import requests
import re  # For space cleanup

st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("🤖 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to query AI API and stream response
def query_ai(question):
    url = "http://127.0.0.1:7860/query/"
    params = {"input_text": question}
    
    with requests.get(url, params=params, stream=True) as response:
        if response.status_code == 200:
            full_response = ""
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    text_chunk = chunk.decode("utf-8")
                    full_response += text_chunk
                    yield full_response  # Streamed response

# Custom CSS for spacing fix
st.markdown("""
    <style>
        .chat-box {
            background-color: #1e1e1e;
            padding: 12px;
            border-radius: 10px;
            margin-top: 5px;
            font-size: 154x;
            font-family: monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.2;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

user_input = st.text_input("Ask a question:", "", key="user_input")
submit_button = st.button("Submit")

if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Placeholder for streaming
    response_container = st.empty()
    full_response = ""

    with st.spinner("🤖 AI is thinking..."):
        for chunk in query_ai(user_input):
            full_response = chunk
            response_container.markdown(f'<div class="chat-box">{full_response}</div>', unsafe_allow_html=True)

    response_container.empty()  # Hides the streamed "Thinking" response after completion

    # Extract refined answer after "</think>"
    if "</think>" in full_response:
        refined_response = full_response.split("</think>", 1)[-1].strip()
    else:
        refined_response = full_response  # Fallback if </think> is missing

    # Remove extra newlines and excessive spaces
    refined_response = re.sub(r'\n\s*\n', '\n', refined_response.strip())

    # Expandable AI Thought Process Box
    with st.expander("🤖 AI's Thought Process (Click to Expand)"):
        st.markdown(f'<div class="chat-box">{full_response}</div>', unsafe_allow_html=True)

    # Display refined answer with clean formatting
    st.write("Answer:")
    st.markdown(refined_response, unsafe_allow_html=True)
