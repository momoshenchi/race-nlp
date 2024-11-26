import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from modelscope.msdatasets import MsDataset
import torch

def load_test_data(test_file):
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    return test_data

def batch_encode(model, sentences, batch_size=32):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        emb = model.encode(batch, convert_to_tensor=True)
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)

def compute_similarity_matrix(test_embeddings, corpus_embeddings):
    # 计算余弦相似度矩阵 [num_test, num_corpus]
    return torch.nn.functional.cosine_similarity(
        test_embeddings.unsqueeze(1),  # [num_test, 1, dim]
        corpus_embeddings.unsqueeze(0),  # [1, num_corpus, dim]
        dim=2
    )

def main():
    # 加载模型
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 加载测试数据
    test_data = load_test_data('eval_only_query.jsonl')
    
    # 预编码所有测试查询
    test_queries = [item['query'] for item in test_data]
    test_embeddings = batch_encode(model, test_queries)
    
    # 相似度阈值
    threshold = 0.7
    
    # 打开文件准备写入
    log_file = open('similarity_logs.txt', 'w', encoding='utf-8')
    similar_file = open('similar_pairs.jsonl', 'w', encoding='utf-8')
    
    # 加载训练集并批量处理
    ds = MsDataset.load('BAAI/IndustryCorpus2_tourism_geography',split="train")
    batch_size = 1000
    corpus_texts = []
    
    # 使用tqdm显示处理进度
    for train_item in tqdm(ds, desc="Processing training data"):
        train_text = train_item['text']  # 根据实际数据集字段调整
        corpus_texts.append(train_text)
        
        # 当收集够一个批次时进行处理
        if len(corpus_texts) >= batch_size:
            # 编码当前批次的训练文本
            corpus_embeddings = batch_encode(model, corpus_texts)
            
            # 计算相似度矩阵 [num_test, batch_size]
            similarity_matrix = compute_similarity_matrix(test_embeddings, corpus_embeddings)
            
            # 处理相似度结果
            for test_idx in range(len(test_queries)):
                for corpus_idx in range(len(corpus_texts)):
                    similarity = similarity_matrix[test_idx, corpus_idx].item()
                    
                    # 记录日志
                    log_entry = {
                        'test_query': test_queries[test_idx],
                        'train_text': corpus_texts[corpus_idx],
                        'similarity': similarity
                    }
                    log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                    
                    # 如果相似度超过阈值，直接写入文件
                    if similarity > threshold:
                        similar_entry = {
                            'train_text': corpus_texts[corpus_idx],
                            'similarity': similarity
                        }
                        similar_file.write(json.dumps(similar_entry, ensure_ascii=False) + '\n')
            
            # 清空当前批次
            corpus_texts = []
    
    # 处理最后一个不完整的批次
    if corpus_texts:
        corpus_embeddings = batch_encode(model, corpus_texts)
        similarity_matrix = compute_similarity_matrix(test_embeddings, corpus_embeddings)
        
        for test_idx in range(len(test_queries)):
            for corpus_idx in range(len(corpus_texts)):
                similarity = similarity_matrix[test_idx, corpus_idx].item()
                
                log_entry = {
                    'test_query': test_queries[test_idx],
                    'train_text': corpus_texts[corpus_idx],
                    'similarity': similarity
                }
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
                if similarity > threshold:
                    similar_entry = {
                        'train_text': corpus_texts[corpus_idx],
                        'similarity': similarity
                    }
                    similar_file.write(json.dumps(similar_entry, ensure_ascii=False) + '\n')
    
    # 关闭文件
    log_file.close()
    similar_file.close()

if __name__ == '__main__':
    main()