import os
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "maibelai"

class VectorStoreManager:
    def __init__(self):
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        Pinecone(api_key=PINECONE_API_KEY)

    def __get_vector_store(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    def retrieve_from_vector_store(self, query: str, top_k: int):
        vector_store = self.__get_vector_store()
        results = vector_store.similarity_search(query, k=top_k)
        return results
