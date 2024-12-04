import json
import hashlib
import re
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Qwen/Qwen2.5-7B-Instruct"
log_file = "run_log_qwen25_tot_version3.jsonl"  # 日志
ans_file = "model_logits_qwen25_tot_version3.jsonl"  # 推理结果
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype="auto")
params = {
            "max_new_tokens": 4096,
            "top_p": 0.95,
            "temperature": 0.8,
            "do_sample": True
        }
def llm(messages):
    input_id = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_id], return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **params)
    prompt_len = len(inputs['input_ids'][0])
    output_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return input_id,response
def self_verification(response, inp,types):
    """
    对每个回答进行自我验证、思考和纠正。
    """
    # verification_prompt = f"""
    # 你是一位专业且有帮助的AI助手。下面是一个问题及其回答，请对以下回答进行自我验证并进行必要的修正，确保回答符合以下标准：
#     1. 准确性：检查回答中是否有错误或误导性信息，如果有，请纠正它。
#     2. 完整性：检查回答是否遗漏了问题的关键信息，如果有，请补充遗漏的部分。
#     3. 逻辑性：验证回答的逻辑是否清晰连贯，是否存在矛盾或不合理之处，若有，请修正。
#     4. 专业性：检查回答中是否使用了合适的专业术语，避免使用不恰当的词汇。
#     5.格式:是否遵守了相关格式,如下

#     **回答示例：**

# **示例问题 1：**  
# 问题：如何监控企业贷款的风险状况？
# 选项：A:通过定期或不定期的财务报告，对企业的财务状况进行监控，一旦出现风险信号，立即采取措施。\nB:对企业的信用状况进行详细的评估，包括企业的还款能力、经营状况、财务状况等。\nC:根据企业的风险状况，设定一定的风险储备金，以应对可能出现的风险。\nD:通过培训和教育，提高企业对贷款风险的认识，引导企业合理使用贷款。


# **AI回答：**  
# 正确答案是A：通过定期或不定期的财务报告，对企业的财务状况进行监控，一旦出现风险信号，立即采取措施。
# 解析：
# A. 正确，定期或不定期地对企业财务报告进行审查，可以及时发现企业在经营过程中的潜在问题，如应收账款增加、现金流减少、利润下降等，这些都可能是企业面临风险的信号。一旦发现问题，金融机构可以及时采取相应措施，如提高贷款利率、要求企业提供额外担保或限制贷款额度，从而有效管理信贷风险。
# B. 虽然对企业的信用状况进行详细的评估是风险管理的重要组成部分，但不是直接监控企业贷款风险状况的方式。详细的信用评估有助于全面了解企业的还款能力和经营状况，为风险监控提供背景信息。在动态监控方面，定期审查财务报告更具时效性。
# C. 风险储备金制度有助于企业或金融机构在面对突发情况时能够有足够的资金缓冲来应对可能产生的损失。这是一种风险管理和资本充足性的工具，但同样不是直接的贷款风险监控手段。
# D. 提高企业对贷款风险的认识确实在一定程度上有助于降低风险，但这更多是在事前教育和预防层面的做法，而不是实时风险监控的具体操作。引导企业合理使用贷款也属于风险管理的范畴，而不是直接监控流程的一部分。\n\n

# 综上所述，定期或不定期地审查企业的财务报告是更为直接有效的风险监控手段。其他选项虽重要，但它们更多关注的是风险预防或风险管理策略，而非实时的风险监控过程。


# **示例问题 2：**  
# 问题：How can a mobile application incorporating emotional finance strategies assist individuals in making better financial decisions and managing their emotions in the field of behavioral finance?
# **AI回答：** 
# A mobile application that incorporates emotional finance strategies can significantly assist individuals in making better financial decisions and managing their emotions by addressing the psychological and emotional factors that influence financial behavior. Here's how such an application can achieve this:
# 1. **Emotional Awareness and Tracking**:
# The app can help users become more aware of their emotional states when making financial decisions. By incorporating features like mood tracking or prompting users to note their feelings during transactions, the app encourages mindfulness about how emotions like fear, greed, or anxiety may impact their choices.
# 2. **Educational Content on Behavioral Finance**:
# Providing accessible educational resources about common cognitive biases and emotional pitfalls in financial decision-making empowers users with knowledge. Understanding concepts like loss aversion, overconfidence, or herd behavior can help users recognize and mitigate these biases in their own actions.
# 3. **Personalized Insights and Feedback**:
# Utilizing data analytics, the app can monitor spending patterns, investment behaviors, and emotional inputs to offer personalized feedback. For instance, if the app detects impulsive spending during periods of stress, it can alert the user and provide strategies to cope with these triggers.
# 4. **Cognitive Behavioral Techniques (CBT)**:
# Integrating CBT exercises can help users challenge and change unhelpful financial thought patterns. The app might include activities that address negative self-talk about money, help reframe financial setbacks, and promote a growth mindset towards personal finance.
# 5. **Mindfulness and Stress-Reduction Tools**:
# Including features like guided meditations, deep-breathing exercises, or quick mindfulness activities can help users manage stress and emotional responses that often lead to poor financial decisions. Managing stress can reduce impulsivity and enhance decision-making clarity.

# By addressing both the rational and emotional components of financial decision-making, the app serves as a comprehensive tool that not only manages finances but also promotes emotional well-being. It bridges the gap between financial advice and psychological support, recognizing that money management is as much about emotions and behavior as it is about numbers.


#     """
    with open("verification_prompt.txt", "r") as f:
        verification_prompt=f.read()
    user_prompt = f"""
    原始回答：
    {response}
    
    请对该回答进行自我修正，确保其满足上述标准，并返回修正后的版本。直接返回修改后的答案，不要使用"修改后的答案"，"修正后的答案"等词语开头。
    """
    verification_message=[{"role": "system", "content": verification_prompt},
                         {"role": "user", "content":format_inp(inp,types)+"\n\n"+user_prompt}]

    inp_final, corrected_response = llm(verification_message)
    print(corrected_response)
    return inp_final,corrected_response

FINANCE_PROMPT = """
你是一位专业且有帮助的AI助手，专注于解答与金融、经济相关的各种问题，包括选择题和开放性问题。你的回答需要科学严谨、逻辑清晰、语言流畅，并根据问题类型提供详细且内容丰富的回答。具体要求如下：

1. **选择题格式：**
   - **明确指出正确答案：** 在回答开头清晰给出正确答案。
   - **详细解释理由：** 结合逻辑推导和背景知识，确保答案的合理性和说服力。
   - **逐一分析其他选项：** 说明每个选项的优缺点，特别是为什么其他选项不符合当前问题的要求。

2. **开放性问题：**
   - **提供全面且逻辑缜密的回答：** 结合背景知识和实际案例，从多个角度进行详细分析，尽可能满足用户对问题的深度和广度需求。
   - **引用权威来源：** 如果相关，引用实际案例、成功经验或科学研究支持回答。
   - **提出实用建议：** 对于有实际应用的主题，如技术实现或方案规划，描述步骤详尽，技术原理清晰，尽量使用用户易理解的语言表达。

3. **语言风格和内容要求：**
   - 保持语言专业但易于理解，避免使用过多晦涩术语，但确保科学表达准确。
   - 逻辑连贯，尽量丰富字数，充分覆盖问题的所有相关方面。
   - 字数尽量丰富，以充分满足用户对深度和广度的期望。

4. **回答示例：**

**示例问题 1：**  
问题：如何监控企业贷款的风险状况？
选项：A:通过定期或不定期的财务报告，对企业的财务状况进行监控，一旦出现风险信号，立即采取措施。\nB:对企业的信用状况进行详细的评估，包括企业的还款能力、经营状况、财务状况等。\nC:根据企业的风险状况，设定一定的风险储备金，以应对可能出现的风险。\nD:通过培训和教育，提高企业对贷款风险的认识，引导企业合理使用贷款。

**AI回答：**  
正确答案是A：通过定期或不定期的财务报告，对企业的财务状况进行监控，一旦出现风险信号，立即采取措施。
解析：
A. 正确，定期或不定期地对企业财务报告进行审查，可以及时发现企业在经营过程中的潜在问题，如应收账款增加、现金流减少、利润下降等，这些都可能是企业面临风险的信号。一旦发现问题，金融机构可以及时采取相应措施，如提高贷款利率、要求企业提供额外担保或限制贷款额度，从而有效管理信贷风险。
B. 虽然对企业的信用状况进行详细的评估是风险管理的重要组成部分，但不是直接监控企业贷款风险状况的方式。详细的信用评估有助于全面了解企业的还款能力和经营状况，为风险监控提供背景信息。在动态监控方面，定期审查财务报告更具时效性。
C. 风险储备金制度有助于企业或金融机构在面对突发情况时能够有足够的资金缓冲来应对可能产生的损失。这是一种风险管理和资本充足性的工具，但同样不是直接的贷款风险监控手段。
D. 提高企业对贷款风险的认识确实在一定程度上有助于降低风险，但这更多是在事前教育和预防层面的做法，而不是实时风险监控的具体操作。引导企业合理使用贷款也属于风险管理的范畴，而不是直接监控流程的一部分。\n\n

综上所述，定期或不定期地审查企业的财务报告是更为直接有效的风险监控手段。其他选项虽重要，但它们更多关注的是风险预防或风险管理策略，而非实时的风险监控过程。


**示例问题 2：**  
问题：How can a mobile application incorporating emotional finance strategies assist individuals in making better financial decisions and managing their emotions in the field of behavioral finance?

**AI回答：** 
A mobile application that incorporates emotional finance strategies can significantly assist individuals in making better financial decisions and managing their emotions by addressing the psychological and emotional factors that influence financial behavior. Here's how such an application can achieve this:
1. **Emotional Awareness and Tracking**:
The app can help users become more aware of their emotional states when making financial decisions. By incorporating features like mood tracking or prompting users to note their feelings during transactions, the app encourages mindfulness about how emotions like fear, greed, or anxiety may impact their choices.
2. **Educational Content on Behavioral Finance**:
Providing accessible educational resources about common cognitive biases and emotional pitfalls in financial decision-making empowers users with knowledge. Understanding concepts like loss aversion, overconfidence, or herd behavior can help users recognize and mitigate these biases in their own actions.
3. **Personalized Insights and Feedback**:
Utilizing data analytics, the app can monitor spending patterns, investment behaviors, and emotional inputs to offer personalized feedback. For instance, if the app detects impulsive spending during periods of stress, it can alert the user and provide strategies to cope with these triggers.
4. **Cognitive Behavioral Techniques (CBT)**:
Integrating CBT exercises can help users challenge and change unhelpful financial thought patterns. The app might include activities that address negative self-talk about money, help reframe financial setbacks, and promote a growth mindset towards personal finance.
5. **Mindfulness and Stress-Reduction Tools**:
Including features like guided meditations, deep-breathing exercises, or quick mindfulness activities can help users manage stress and emotional responses that often lead to poor financial decisions. Managing stress can reduce impulsivity and enhance decision-making clarity.

By addressing both the rational and emotional components of financial decision-making, the app serves as a comprehensive tool that not only manages finances but also promotes emotional well-being. It bridges the gap between financial advice and psychological support, recognizing that money management is as much about emotions and behavior as it is about numbers.

"""

def format_inp(inp,types):
    if types == "objective":
        return inp
    else:
        return "请回答以下开放式问题: "+inp

def get_best_answer(inp,types):
    messages = [
                {"role": "system", "content": FINANCE_PROMPT  },
                {"role": "user", "content": format_inp(inp,types)}
            ]
    # 获取5次回答
    responses = []
    ver_inps=[]
    for _ in range(5):
        inp_final,res=llm(messages)
        ver_inp,verified_res = self_verification(res, inp,types)  # 对每个回答进行自我验证
        responses.append(verified_res)
        ver_inps.append(ver_inp)
#     ranking_prompt = """下面是5个回答,请你按照以下标准进行排序,你只需要输出最终的JSON结果即可, 不需要输出解释和说明:
#     1. 准确性 - 内容是否严谨、正确
#     2. 完整性 - 是否完整回答了问题的所有方面
#     3. 逻辑性 - 论述是否清晰、连贯
#     4. 专业性 - 是否使用了恰当的专业术语
#     5. 格式 - 是否符合回答格式要求

#     **回答示例：**

# **示例问题 1：**  
# 问题：如何监控企业贷款的风险状况？
# 选项：A:通过定期或不定期的财务报告，对企业的财务状况进行监控，一旦出现风险信号，立即采取措施。\nB:对企业的信用状况进行详细的评估，包括企业的还款能力、经营状况、财务状况等。\nC:根据企业的风险状况，设定一定的风险储备金，以应对可能出现的风险。\nD:通过培训和教育，提高企业对贷款风险的认识，引导企业合理使用贷款。


# **AI回答：**  
# 正确答案是A：通过定期或不定期的财务报告，对企业的财务状况进行监控，一旦出现风险信号，立即采取措施。
# 解析：
# A. 正确，定期或不定期地对企业财务报告进行审查，可以及时发现企业在经营过程中的潜在问题，如应收账款增加、现金流减少、利润下降等，这些都可能是企业面临风险的信号。一旦发现问题，金融机构可以及时采取相应措施，如提高贷款利率、要求企业提供额外担保或限制贷款额度，从而有效管理信贷风险。
# B. 虽然对企业的信用状况进行详细的评估是风险管理的重要组成部分，但不是直接监控企业贷款风险状况的方式。详细的信用评估有助于全面了解企业的还款能力和经营状况，为风险监控提供背景信息。在动态监控方面，定期审查财务报告更具时效性。
# C. 风险储备金制度有助于企业或金融机构在面对突发情况时能够有足够的资金缓冲来应对可能产生的损失。这是一种风险管理和资本充足性的工具，但同样不是直接的贷款风险监控手段。
# D. 提高企业对贷款风险的认识确实在一定程度上有助于降低风险，但这更多是在事前教育和预防层面的做法，而不是实时风险监控的具体操作。引导企业合理使用贷款也属于风险管理的范畴，而不是直接监控流程的一部分。\n\n

# 综上所述，定期或不定期地审查企业的财务报告是更为直接有效的风险监控手段。其他选项虽重要，但它们更多关注的是风险预防或风险管理策略，而非实时的风险监控过程。


# **示例问题 2：**  
# 问题：How can a mobile application incorporating emotional finance strategies assist individuals in making better financial decisions and managing their emotions in the field of behavioral finance?
# **AI回答：** 
# A mobile application that incorporates emotional finance strategies can significantly assist individuals in making better financial decisions and managing their emotions by addressing the psychological and emotional factors that influence financial behavior. Here's how such an application can achieve this:
# 1. **Emotional Awareness and Tracking**:
# The app can help users become more aware of their emotional states when making financial decisions. By incorporating features like mood tracking or prompting users to note their feelings during transactions, the app encourages mindfulness about how emotions like fear, greed, or anxiety may impact their choices.
# 2. **Educational Content on Behavioral Finance**:
# Providing accessible educational resources about common cognitive biases and emotional pitfalls in financial decision-making empowers users with knowledge. Understanding concepts like loss aversion, overconfidence, or herd behavior can help users recognize and mitigate these biases in their own actions.
# 3. **Personalized Insights and Feedback**:
# Utilizing data analytics, the app can monitor spending patterns, investment behaviors, and emotional inputs to offer personalized feedback. For instance, if the app detects impulsive spending during periods of stress, it can alert the user and provide strategies to cope with these triggers.
# 4. **Cognitive Behavioral Techniques (CBT)**:
# Integrating CBT exercises can help users challenge and change unhelpful financial thought patterns. The app might include activities that address negative self-talk about money, help reframe financial setbacks, and promote a growth mindset towards personal finance.
# 5. **Mindfulness and Stress-Reduction Tools**:
# Including features like guided meditations, deep-breathing exercises, or quick mindfulness activities can help users manage stress and emotional responses that often lead to poor financial decisions. Managing stress can reduce impulsivity and enhance decision-making clarity.

# By addressing both the rational and emotional components of financial decision-making, the app serves as a comprehensive tool that not only manages finances but also promotes emotional well-being. It bridges the gap between financial advice and psychological support, recognizing that money management is as much about emotions and behavior as it is about numbers.

    
#     请以JSON格式输出排序结果,格式如下:
#     {
#         "rankings": [排名第1的回答序号, 排名第2的回答序号, ..., 排名第5的回答序号]
#     }
#     (序号范围为1-5)
#     """
    # 构建排序prompt,要求JSON格式输出
    with open("ranking_prompt.txt", "r") as f:
        ranking_prompt=f.read()

    rank_messages =  [{"role": "system", "content": ranking_prompt},
                {"role": "user", "content": format_inp(inp,types) +"\n\n"+"五个回答如下:"+ "\n\n".join([f"回答{i+1}:\n{resp}" for i, resp in enumerate(responses)])}]
    
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
        inp_select=ver_inps[rankings[0]]
        # 按排名顺序重排responses
        ranked_responses = [responses[i] for i in rankings]
        return inp_final+""+inp_select+""+inp_next,ranked_responses[0]
           
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
        
        inp_select=ver_inps[rankings[0]]
        # 按排名顺序重排responses
        ranked_responses = [responses[i] for i in rankings]
        return inp_final+""+inp_select+""+inp_next,ranked_responses[0]

def generate_response(inp_list, unique_id_list,typelist):
    try:
        responses = []
        run_infos = []
        for i,inp in enumerate(inp_list):
            types=typelist[i]
            prompt, response = get_best_answer(inp,types)
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

    responses, run_infos = generate_response(prompts, unique_ids,query_types)

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