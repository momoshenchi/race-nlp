from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import json
from tqdm import tqdm
import hashlib
import torch
from torch.utils.data import DataLoader, Dataset

device = "cuda:3"  # the device to load the model onto
model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
test_file = "eval_only_query.jsonl"

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 构建带有时间戳的日志文件和答案文件名
log_file = f"run_log_qwen25_{timestamp}.jsonl"
ans_file = f"model_logits_qwen25_{timestamp}.jsonl"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype="auto", device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')

# 启用flash attention（如果支持）
# if hasattr(model.config, "use_flash_attention"):
#     model.config.use_flash_attention = True

params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "do_sample": True,
    "max_new_tokens": 8192,
    "use_cache": True,  # 启用KV缓存
}


TOURISM_PROMPT = """
你是一位专业且有帮助的AI助手，专门回答与旅游、地理、交通相关的各种问题，包括选择题和开放式问题。你的回答应确保内容严谨、准确，语言与问题的主要语言保持一致，并提供详细的逻辑支持。你的回答需要尽可能详细、内容丰富，并在相关主题上给出全面的信息，字数尽量多以满足用户对深度和广度的需求。具体指导如下：

1. **选择题格式：**
   - **明确给出正确答案**，并在开头清晰标注。
   - **详细解释理由**，结合逻辑推导和背景知识，确保答案的合理性和说服力。
   - **逐一分析其他选项**，指出其优点、不足或与正确答案的差异，并说明为什么它们不适合当前问题。

2. **开放式问题：**
   - 提供**详尽且逻辑缜密**的回答，结合背景知识、技术工具、历史文化等多角度阐述。
   - 如果相关，引用实际案例、成功经验或科学研究支持回答。
   - 对于有实际应用的主题，如技术实现或方案规划，描述步骤详尽，技术原理清晰，尽量使用用户易理解的语言表达。

3. **语言风格和内容要求：**
   - 语言与问题主要语言一致，保证清晰流畅。
   - 确保回答逻辑连贯、全面覆盖问题的各个维度。
   - 字数尽量丰富，以充分满足用户对深度和广度的期望。

4. **回答示例：**

用户问题：*如何利用地理数据可视化技术方便用户查找和规划旅行路线？*

AI回答：
1. **地图服务与API整合：**
   - 使用主流地图服务提供商的API，如Google Maps API、高德地图API或百度地图API。这些平台提供丰富的地理数据，支持动态交通信息的展示和路线规划。
   - 结合用户行为数据和兴趣点推荐功能，通过地理信息可视化显示适合用户的路线和地点。

2. **多模式交通分析：**
   - 借助地理数据可视化，展示多种出行方式（如驾车、步行、公共交通）的时间和成本对比。
   - 提供动态切换功能，例如使用热力图显示交通流量，帮助用户避开拥堵。

3. **交互式地图设计：**
   - 设计用户友好的交互界面，支持缩放、旋转和图层切换功能，让用户可以从宏观视角观察城市交通，也能查看微观细节如停车位分布。

4. **基于数据驱动的个性化推荐：**
   - 结合用户历史行为数据，推荐用户可能感兴趣的旅游景点、酒店和餐饮，并根据用户选择的路线提供实时调整。
   - 利用天气和实时活动信息，通过地图标注向用户提醒关键动态。

5. **数据分析支持与技术实现：**
   - 提供算法支持，例如Dijkstra算法或A*算法，用于快速计算最短路径。
   - 整合地理数据库（如PostGIS）和可视化库（如D3.js、Leaflet）构建系统。

6. **实际案例支持：**
   - 引用现有的成功案例，例如Uber的动态路径规划和Citymapper的多模式交通解决方案，展示技术如何在实际场景中帮助用户。

通过以上详细回答，帮助用户全面理解如何运用地理数据可视化技术来优化旅行规划过程。
"""

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


class TestDataset(Dataset):
    def __init__(self, file_path):
        self.test_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.test_data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        item = self.test_data[idx]
        prompt = item["query"]
        query_type = item["query_type"]
        return prompt, query_type


def collate_fn(batch):
    prompts, query_types = zip(*batch)
    messages = [
        [
            {
                "role": "system",
                "content": FINANCE_PROMPT,
            },
            {"role": "user", "content": prompt if query_types[i] == "objective" else "请回答以下开放式问题: "+prompt},
        ]
        for i,prompt in enumerate(prompts)
    ]
    texts = [
        tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]
    model_inputs = tokenizer(
        texts, padding=True, return_tensors="pt"
    ).to(device)
    return model_inputs, prompts, texts,query_types


def generate_response(model_inputs):
    generated_ids = model.generate(**model_inputs, **params)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


def generate_unique_code(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode("utf-8"))
    unique_code = sha256_hash.hexdigest()
    return unique_code


# 创建数据集和数据加载器
test_dataset = TestDataset(test_file)
test_loader = DataLoader(
    test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
)

for batch_index, (model_inputs, prompts,texts, query_types) in enumerate(
    tqdm(test_loader, desc="Processing Data")
):
    responses = generate_response(model_inputs)

    for i in range(len(prompts)):
        prompt = prompts[i]
        response = responses[i]
        query_type = query_types[i]
        text=texts[i]
        # 保存到答案文件
        answer_info = {
            "query": prompt,
            "query_type": query_type,
            "answer": response,
        }

        with open(ans_file, "a") as fw:
            fw.write(json.dumps(answer_info, ensure_ascii=False) + "\n")

        # 保存日志文件
        run_info = {
            "unique_id": generate_unique_code(prompt),
            "model_inp": text,
            "gen_params": params,
            "answer": response,
        }

        # 保存运行日志
        with open(log_file, "a") as fw:
            fw.write(json.dumps(run_info, ensure_ascii=False) + "\n")

    # 清空显存
    # torch.cuda.empty_cache()
