import os
import requests
import chromadb
from embedder import CustomEmbedder

def query(prompt):
    API_TOKEN = os.environ["API_TOKEN"] #Set a API_TOKEN environment variable before running
    API_URL = "https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud/" #Add a URL for a model of your choosing
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    payload = {
        "inputs": prompt,
        "parameters": { #Try and experiment with the parameters
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": False,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text']

def as_retriever(collection, query):
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    return results['documents']

def main():
    custom_embedder = CustomEmbedder()
    client = chromadb.PersistentClient(path="../testdb")
    collection = client.get_collection(name="test_collection", embedding_function=custom_embedder)

    question = "Do I have a pet?"

    context = as_retriever(collection, question)
    prompt = f"""Use the following context to answer the question at the end. Stop when you've answered the question. Do not generate any more than that.

    {context}

    Question: {question}
    """

    print(query(prompt))


if __name__ == "__main__":
    main()