from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
import os
import pandas as pd
from ecommbot.data_converter import dataconveter

load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")

# Create embeddings object using an open-source model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def ingestdata():
    vstore = AstraDBVectorStore(
            embedding=embedding,
            collection_name="chatbotecomm",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )
    
    # Attempt a dummy query to check if data exists
    try:
        test_results = vstore.similarity_search("OnePlus Bullets")
        if test_results:  # If data exists, we should get some results
            print("** Data already exists in AstraDB; skipping ingestion. **")
            return vstore, None
    except Exception as e:
        print("Error during data existence check:", e)

    # If no data was found or an error occurred, perform data ingestion
    print("** No existing data found; ingesting new documents. **")
    docs = dataconveter()
    inserted_ids = vstore.add_documents(docs)
    return vstore, inserted_ids

if __name__=='__main__':
    vstore,inserted_ids = ingestdata()
    if inserted_ids is not None:
        print(f"\nInserted {len(inserted_ids)} documents.")
    
    results = vstore.similarity_search("can you tell me the low budget sound basshead.")
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")