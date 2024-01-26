import chromadb

def main():
    dataset = [
        "My cat is named Francis.",
        "I want to visit Italy.",
        "I need to download more RAM."
    ]
    chroma_client = chromadb.PersistentClient(path="../testdb")
    chroma_client.delete_collection("test_collection")
    collection = chroma_client.create_collection(name="test_collection")
    
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