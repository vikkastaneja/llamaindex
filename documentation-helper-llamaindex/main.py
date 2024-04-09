import os
from dotenv import load_dotenv
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

load_dotenv()

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore

if __name__ == '__main__':
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(name=os.getenv('PINECONE_INDEX'))
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    Settings.callback_manager = callback_manager

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query = 'What is a llamaindex query engine?'
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)