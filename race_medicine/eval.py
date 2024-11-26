import json
import hashlib
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2-7B-Instruct"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype="auto")

def generate_response(inp_list, unique_id_list, max_new_tokens=2048):

    # 与训练使用的template保持一致
    try:
        prompts = [f"User: {inp}\nAssistant: " for inp in inp_list]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        params = {
            "max_new_tokens": max_new_tokens,
            "top_p": 0.95,
            "temperature": 0.8,
            "do_sample": True
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **params)
        responses = []
        run_infos = []
        for i in range(len(inp_list)):
            prompt_len = len(inputs['input_ids'][i])
            output_ids = outputs[i][prompt_len:]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            responses.append(response)
            run_info = {
                "unique_id": unique_id_list[i],
                "model_inp": prompts[i],
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

log_file = "run_log.jsonl"  # 日志
ans_file = "model_logits.jsonl"  # 推理结果

# 加载测试数据
test_data = load_test_data('eval_only_query.jsonl')
print("test")

batch_size = 8
for i in range(0, len(test_data), batch_size):
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
        with open(log_file, "a") as fw:
            fw.write(json.dumps(run_info, ensure_ascii=False) + "\n")

        # 保存输出结果
        answer_info = {
            "query": prompts[j],
            "query_type": query_types[j],
            "answer": response
        }

        with open(ans_file, "a") as fw:
            fw.write(json.dumps(answer_info, ensure_ascii=False) + "\n")
        print(f"answer_info: {answer_info}")
