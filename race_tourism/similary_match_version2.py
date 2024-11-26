import json
from tqdm import tqdm
import numpy as np
from modelscope.msdatasets import MsDataset
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_test_data(test_file):
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    return test_data

def main():
    print("start")
    # Initialize the embedding model
    model_kwarg = {'device': 'cuda'}
    embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-multilingual-gemma2',model_kwargs= model_kwarg)

    # Load test data
    test_data = load_test_data('eval_only_query.jsonl')
    test_queries = [item['query'] for item in test_data]
    test_embeddings = embedding_model.embed_documents(test_queries)

    # Similarity threshold
    threshold = 0.7

    # Open file to write similar pairs
    similar_file = open('similar_pairs.jsonl', 'w', encoding='utf-8')

    # Load training data and build Faiss index
    ds = MsDataset.load('BAAI/IndustryCorpus2_tourism_geography',split="train")
    corpus_texts = []
    corpus_embeddings = []

    # Build the index in batches to handle large datasets
    batch_size = 100000  # Adjust based on your memory capacity
    index = None  # Initialize the Faiss index

    for idx, train_item in enumerate(tqdm(ds, desc="Loading and indexing corpus")):
        train_text = train_item['text']  # Adjust if the field is different
        corpus_texts.append(train_text)

        # When batch is full or it's the last item, process the batch
        if len(corpus_texts) >= batch_size or idx == len(ds) - 1:

            if index is None:
                # Initialize the Faiss index
                index = FAISS.from_texts(
                texts=corpus_texts,
                embedding=embedding_model
            )
            else:
            # Add to existing Faiss index
                index.add_texts(corpus_texts)

            # Clear the lists for the next batch
            corpus_texts = []

    # Search for similar texts for each test query
    for test_embedding in tqdm( test_embeddings, desc="Searching for similar texts", total=len(test_queries)):
        # Query the Faiss index
        similar_docs = index.similarity_search_by_vector(test_embedding, k=10)  # Adjust 'k' as needed

        # Filter results by similarity threshold
        for doc in similar_docs:
            # Compute similarity manually since Faiss returns approximate distances
            similarity = np.dot(test_embedding, embedding_model.embed_documents([doc.page_content])[0]) / \
                         (np.linalg.norm(test_embedding) * np.linalg.norm(embedding_model.embed_documents([doc.page_content])[0]))
            if similarity > threshold:
                similar_entry = {
                    'train_text': doc.page_content,
                    'similarity': similarity
                }
                similar_file.write(json.dumps(similar_entry, ensure_ascii=False) + '\n')

    similar_file.close()

main()
