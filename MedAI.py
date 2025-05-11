import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def main():
    st.set_page_config(page_title="MedAI Chatbot", page_icon="💬", layout="wide")
    st.markdown(
        """
        <style>
        .stChatMessage {
            background-color: #2c3e50;
            color: #ecf0f1;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            display: inline-block;
            max-width: 70%;
        }
        .stChatMessageUser {
            background-color: #1abc9c;
            color: #ecf0f1;
            border-radius: 10px;
            padding: 5px 10px;
            margin: 10px 0;
            display: inline-block;
            max-width: 70%;
            text-align: right;
            float: right;
        }
        .stChatMessageAssistant {
            background-color: #34495e;
            color: #ecf0f1;
            border-radius: 10px;
            padding: 5px 10px;
            margin: 10px 0;
            display: inline-block;
            max-width: 70%;
            text-align: left;
            float: left;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("💬 MedAI Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role_class = 'stChatMessageUser' if message['role'] == 'user' else 'stChatMessageAssistant'
        st.markdown(f"<div class='{role_class}'>{message['content']}</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.markdown(f"<div class='stChatMessageUser'>{prompt}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, say you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Give answer in a proper format.
        Start the answer directly. No small talk.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]
            result_to_show = result

            st.markdown(f"<div class='stChatMessageAssistant'>{result_to_show}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
