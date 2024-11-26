import json
from tqdm import tqdm
import numpy as np
import torch
from modelscope.msdatasets import MsDataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_core.documents import Document
def load_test_data(test_file):
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    return test_data

def embed_in_batches(embedding_model,texts, batch_size=512):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(embedding_model.embed_documents(batch))
    return embeddings

def main():
    
    # Initialize the embedding model
    model_kwarg = {'device': 'cuda'}
    embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs=model_kwarg)

    # Load test data
    test_data = load_test_data('eval_only_query.jsonl')
    test_queries = [item['query'] for item in test_data]
    # test_embeddings = embed_in_batches(embedding_model,test_queries)

    # Similarity threshold
    threshold = 0.7

    # Open file to write similar pairs
    similar_file = open('similar_pairs.jsonl', 'w', encoding='utf-8')

    # Load training data and build Faiss index
    ds = MsDataset.load('BAAI/IndustryCorpus2_tourism_geography', split="train")
    corpus_texts = []
    
    # Build the index in batches
    batch_size = 512  # Adjust based on your memory capacity
    index = None  # Initialize the Faiss index
    print("start")
    for idx, train_item in enumerate(tqdm(ds, desc="Loading and indexing corpus")):
        train_text = train_item['text']  # Adjust if the field is different

        if len(train_text)>2048:
            continue
        document = Document(page_content=train_text, metadata={"id": idx})
        corpus_texts.append(document)
        
        if len(corpus_texts) >= batch_size or idx == len(ds) - 1:
            if index is None:
                index = FAISS.from_documents(documents=corpus_texts, embedding=embedding_model)
                gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index.index)
                index.index = gpu_index
            else:
                index.add_texts(corpus_texts)
            corpus_texts = []

    # Search for similar texts for each test query
    buffer = []
    unique_ids = set()
    for test_querie in tqdm(test_queries, desc="Searching for similar texts", total=len(test_queries)):
        similar_docs = index.similarity_search_with_relevance_scores(test_querie, k=100,score_threshold=0.7)  # Adjust 'k' as needed

        for doc,similarity in similar_docs:
            if similarity > threshold and doc.metadata['id'] not in unique_ids:
                unique_ids.add(doc.metadata['id'])
                buffer.append({
                    'train_text': doc.page_content,
                    'similarity': similarity
                })

        if len(buffer) >= 1000:
            similar_file.writelines([json.dumps(entry, ensure_ascii=False) + '\n' for entry in buffer])
            buffer.clear()

    if buffer:
        similar_file.writelines([json.dumps(entry, ensure_ascii=False) + '\n' for entry in buffer])

    similar_file.close()
    print("done",len(unique_ids))
main()
