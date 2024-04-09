import os

import openai
from dotenv import load_dotenv
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex

def main(url:str)->None:
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()
    response = query_engine.query('summarize UI automation')
    print(response)

if __name__ == '__main__':
    print("Hello World!")
    load_dotenv()
    openai.api_key = os.getenv("OPEN_API_KEY")
    print(f'OPEN_API_KEY: {os.getenv("OPEN_API_KEY")}')
    main(url='https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/ui-automation-overview')