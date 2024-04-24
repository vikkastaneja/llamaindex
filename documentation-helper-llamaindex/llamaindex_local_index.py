from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, ServiceContext
from llama_index.core.storage import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI
import dotenv
import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.readers.file import UnstructuredReader

load_dotenv()
storage_path = os.path.dirname(os.path.abspath(__file__)) + "/" + "llamaindex-index-storage"
input_dir=os.path.dirname(os.path.abspath(__file__)) + "/" + "llamaindex-docs"
documents_path = "./docs-huge"

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm)

@st.cache_resource(show_spinner=False)
def initialize():
    if not os.path.exists(storage_path):
        documents = SimpleDirectoryReader(input_dir, file_extractor={".html": UnstructuredReader()}).load_data()
        index = VectorStoreIndex.from_documents(documents=documents)
        index.storage_context.persist(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context=storage_context)
    return index

index = initialize()

st.title("Ask the Document")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role" : "assistant", "content" : "Ask me a question"}
    ]

chat_engine = index.as_chat_engine(chat_more = "condense_question", verbose=True)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role"  : "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            pprint_response(response=response, show_source=True)
            nodes = [node for node in response.source_nodes]
            total_nodes = len(nodes)
            print(total_nodes)
            # print(nodes)
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f'Source node {i+1}: score={node.score}')
                    # st.write(node.text)

            message = {"role" : "assistant", "content": response.response}
            st.session_state.messages.append(message)

