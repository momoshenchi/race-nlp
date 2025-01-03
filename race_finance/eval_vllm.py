from vllm import LLM, SamplingParams
import json
import hashlib
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
#Memory profiling results: total_gpu_memory=23.64GiB 
# initial_memory_usage=7.80GiB peak_torch_memory=9.74GiB 
# memory_usage_post_profile=7.87GiB non_torch_memory=0.74GiB 
# kv_cache_size=10.80GiB gpu_memory_utilization=0.90
model_name = "Qwen/Qwen2.5-7B-Instruct"
log_file = "run_log_qwen25_vllm_test.jsonl"  # 日志
ans_file = "model_logits_qwen25_vllm_test.jsonl"  # 推理结果
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
# 初始化LLM引擎
#张量并行,GPU数量为2,一张显卡无法运行
engine = LLM(model=model_name, tensor_parallel_size=2)
params = {
            "max_tokens": 4096,
            "top_p": 0.95,
            "temperature": 0.8,
        }
def generate_response(inp_list, unique_id_list):
    try:
        input_ids = []
        for inp in inp_list:
            messages = [
            {"role": "system", "content": "你是一位专业且有帮助的AI助手，专门回答与金融、经济相关的各种问题，包括选择题和开放式问题。对于选择题，请先直接给出正确答案，然后详细说明理由，并逐一分析其他选项的优缺点。你的回答应确保内容严谨、准确，语言应与问题中的主要语言保持一致。"},
            {"role": "user", "content": inp}
                ]
            input_id = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids.append(input_id)
        # 设置采样参数
        sampling_params = SamplingParams(**params)
        # 生成输出
        outputs = engine.generate(input_ids, sampling_params)
        responses = []
        run_infos = []

        for i in tqdm(range(len(inp_list)), desc="Generating responses"):
            response = outputs[i].outputs[0].text
            responses.append(response)
            run_info = {
                "unique_id": unique_id_list[i],
                "model_inp": outputs[i].prompt,
                "gen_params": params
            }
            run_infos.append(run_info)
        return responses, run_infos
    except Exception as e:
        traceback.print_exc()
        return None, ()
def generate_unique_code(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode('utf-8'))
    unique_code = sha256_hash.hexdigest()
    return unique_code

def load_test_data(test_file):
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    return test_data

# 加载测试数据
test_data = load_test_data('eval_only_query.jsonl')
print("test")

batch_size = 16
for i in tqdm(range(0, len(test_data), batch_size), desc="Processing batches"):
    batch_items = test_data[i:i+batch_size]
    prompts = [item['query'] for item in batch_items]
    query_types = [item['query_type'] for item in batch_items]
    unique_ids = [generate_unique_code(prompt) for prompt in prompts]

    responses, run_infos = generate_response(prompts, unique_ids)

    if responses is None:
        continue

    for j in range(len(prompts)):
        response = responses[j]
        run_info = run_infos[j]
        run_info["answer"] = response
        # 保存运行日志
        with open(log_file, "a",encoding='utf-8') as fw:
            fw.write(json.dumps(run_info, ensure_ascii=False) + "\n")

        # 保存输出结果
        answer_info = {
            "query": prompts[j],
            "query_type": query_types[j],
            "answer": response
        }

        with open(ans_file, "a",encoding='utf-8') as fw:
            fw.write(json.dumps(answer_info, ensure_ascii=False) + "\n")
        # print(f"answer_info: {answer_info}")
