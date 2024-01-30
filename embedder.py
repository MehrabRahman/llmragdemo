import os
import chromadb
import requests

from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)

class CustomEmbedder(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        rest_client = requests.Session()
        API_TOKEN = os.environ["API_TOKEN"] #Set a API_TOKEN environment variable before running
        API_URL = "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/"
        response = rest_client.post(
            API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {API_TOKEN}"}
        ).json()
        return response

def main():
    dataset = [
        "My cat is named Francis.",
        "I want to visit Italy.",
        "I need to download more RAM."
    ]
    chroma_client = chromadb.PersistentClient(path="../testdb")

    custom_embedder = CustomEmbedder()
    chroma_client.delete_collection("test_collection")
    collection = chroma_client.create_collection(name="test_collection", embedding_function=custom_embedder)
    
    for i, dataset_id in enumerate(dataset):
        collection.add(
            documents=[dataset[i]],
            metadatas=[{"doc_name": "testdoc"}],
            ids=[str(i)]
        )

    results = collection.query(
        query_texts=["travel"],
        n_results=2
    )

    # collection.update(
    #     ids=["2"],
    #     documents="This is an update",
    #     metadatas=[{"source": "I made it up"}]
    # )

    # print(collection.peek())

    print(results)
    

if __name__ == "__main__":
    main()