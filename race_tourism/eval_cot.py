import json
import hashlib
import re
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Qwen/Qwen2.5-7B-Instruct"
log_file = "run_log_qwen25_tot_version2.jsonl"  # 日志
ans_file = "model_logits_qwen25_tot_version2.jsonl"  # 推理结果
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype="auto")
params = {
            "max_new_tokens": 4096,
            "top_p": 0.95,
            "temperature": 0.9,
            "do_sample": True
        }
def self_verification(response, inp):
    """
    对每个回答进行自我验证、思考和纠正。
    """
    verification_prompt = f"""
    你是一位专业且有帮助的AI助手。下面是一个问题及其回答，请对以下回答进行自我验证并进行必要的修正，确保回答符合以下标准：
    1. 准确性：检查回答中是否有错误或误导性信息，如果有，请纠正它。
    2. 完整性：检查回答是否遗漏了问题的关键信息，如果有，请补充遗漏的部分。
    3. 逻辑性：验证回答的逻辑是否清晰连贯，是否存在矛盾或不合理之处，若有，请修正。
    4. 专业性：检查回答中是否使用了合适的专业术语，避免使用不恰当的词汇。
    """
    
    user_prompt = f"""
    原始回答：
    {response}
    
    请对该回答进行自我修正，确保其满足上述标准，并返回修正后的版本。直接返回修改后的答案，不要使用"修改后的答案"，"修正后的答案"等词语开头。
    """
    verification_prompt=[{"role": "system", "content": verification_prompt},
                         {"role": "user", "content":"问题:\n"+inp+"\n\n"+user_prompt}]
    # 这里的 `llm` 应该是一个调用外部语言模型的函数
    inp_final, corrected_response = llm(verification_prompt)
    print(corrected_response)
    return corrected_response
def llm(messages):
    input_id = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_id], return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **params)
    prompt_len = len(inputs['input_ids'][0])
    output_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return input_id,response
def get_best_answer(inp):
    messages = [
                {"role": "system", "content": "你是一位专业且有帮助的AI助手，专门回答与旅游、地理、交通相关的各种问题，包括选择题和开放式问题。对于选择题，请先直接给出正确答案，然后详细说明理由，并逐一分析其他选项的优缺点。你的回答应确保内容严谨、准确，语言应与问题中的主要语言保持一致。"},
                {"role": "user", "content": inp}
            ]
            # 获取5次回答
    responses = []
    for _ in range(5):
        inp_final,res=llm(messages)
        verified_res = self_verification(res, inp)  # 对每个回答进行自我验证
        responses.append(verified_res)

    ranking_prompt = """下面是5个回答,请你按照以下标准进行排序,你只需要输出最终的JSON结果即可, 不需要输出解释和说明:
    1. 准确性 - 内容是否严谨、正确
    2. 完整性 - 是否完整回答了问题的所有方面
    3. 逻辑性 - 论述是否清晰、连贯
    4. 专业性 - 是否使用了恰当的专业术语
    
    请以JSON格式输出排序结果,格式如下:
    {
        "rankings": [排名第1的回答序号, 排名第2的回答序号, ..., 排名第5的回答序号]
    }
    (序号范围为1-5)
    """
    # 构建排序prompt,要求JSON格式输出
    rank_messages =  [{"role": "system", "content":  "你是一位专业且有帮助的AI助手."+ranking_prompt},
                {"role": "user", "content":"问题:"+inp +"\n\n"+"五个回答如下:"+ "\n\n".join([f"回答{i+1}:\n{resp}" for i, resp in enumerate(responses)])}]
    
    
    # 获取排序结果
    inp_next,ranking_result = llm(rank_messages)
    # 提取最佳回答
    try:
        json_content = re.search(r'```json\n(.*?)\n```', ranking_result, re.DOTALL)
        if not json_content:
            ranking_result2 = json.loads(ranking_result)
        else:
            ranking_result2 = json.loads(json_content.group(1))
        # print("json",ranking_result2)
        rankings = [x - 1 for x in ranking_result2["rankings"]]  # 转换为0-based索引
        
        # 按排名顺序重排responses
        ranked_responses = [responses[i] for i in rankings]
        return inp_final,ranked_responses[0]
           
    except json.JSONDecodeError:
        traceback.print_exc()
        inp_next,ranking_result = llm(rank_messages)
        json_content = re.search(r'```json\n(.*?)\n```', ranking_result, re.DOTALL)
        if not json_content:
            ranking_result2 = json.loads(ranking_result)
        else:
            ranking_result2 = json.loads(json_content.group(1))
        print("json",ranking_result2)
        rankings = [x - 1 for x in ranking_result2["rankings"]]  # 转换为0-based索引
        
        # 按排名顺序重排responses
        ranked_responses = [responses[i] for i in rankings]
        # 如果JSON解析失败,返回默认结果
        return inp_final,ranked_responses[0]

def generate_response(inp_list, unique_id_list):
    try:
        responses = []
        run_infos = []
        for i,inp in enumerate(inp_list):
            prompt, response = get_best_answer(inp)
            print("response",response)
            run_info = {
                "unique_id": unique_id_list[i],
                "model_inp": prompt,
                "gen_params": params
            }
            responses.append(response)
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