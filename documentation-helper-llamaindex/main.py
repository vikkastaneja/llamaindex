import os
from dotenv import load_dotenv
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.openai import OpenAI

load_dotenv()
import streamlit as st
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore

@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(name=os.getenv("PINECONE_INDEX"))
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    Settings.callback_manager = callback_manager

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


index = get_index()
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, verbose=True, llm=llm
    )

st.set_page_config(
    page_title="Chat with Llamaindex docs, powered by Llama Index",
    layout="centered",
    page_icon="🦙",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with llamaindex docs, 🦙")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex's open source python library",
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)

            nodes = [node for node in response.source_nodes]
            total_nodes = len(nodes)
            print(total_nodes)
            # print(nodes)
            for col, node in zip(st.columns(len(nodes)), nodes):
                with col:
                    st.header(f'Source node: score={node.score}')

            message = {'role': 'assistant', 'content': response.response}

            st.session_state.messages.append(message)
