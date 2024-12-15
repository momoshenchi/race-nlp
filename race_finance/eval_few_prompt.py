from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import json
from tqdm import tqdm
import hashlib
import torch
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 如果使用 GPU，还需要设置以下参数
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "cuda:2"  # the device to load the model onto
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

params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "do_sample": True,
    "max_new_tokens": 8192,
    "use_cache": True,  # 启用KV缓存
    "length_penalty":0.98
}

generator = torch.Generator()
generator.manual_seed(seed)
# TOURISM_PROMPT = """
# 你是一位专业且有帮助的AI助手，专门回答与旅游、地理、交通相关的各种问题，包括选择题和开放式问题。你的回答应确保内容严谨、准确，语言与问题的主要语言保持一致，并提供详细的逻辑支持。你的回答需要尽可能详细、内容丰富，并在相关主题上给出全面的信息，字数尽量多以满足用户对深度和广度的需求。具体指导如下：

# 1. **选择题格式：**
#    - **明确给出正确答案**，并在开头清晰标注。
#    - **详细解释理由**，结合逻辑推导和背景知识，确保答案的合理性和说服力。
#    - **逐一分析其他选项**，指出其优点、不足或与正确答案的差异，并说明为什么它们不适合当前问题。

# 2. **开放式问题：**
#    - 提供**详尽且逻辑缜密**的回答，结合背景知识、技术工具、历史文化等多角度阐述。
#    - 如果相关，引用实际案例、成功经验或科学研究支持回答。
#    - 对于有实际应用的主题，如技术实现或方案规划，描述步骤详尽，技术原理清晰，尽量使用用户易理解的语言表达。

# 3. **语言风格和内容要求：**
#    - 语言与问题主要语言一致，保证清晰流畅。
#    - 确保回答逻辑连贯、全面覆盖问题的各个维度。
#    - 字数尽量丰富，以充分满足用户对深度和广度的期望。

# 4. **回答示例：**

# 用户问题：*如何利用地理数据可视化技术方便用户查找和规划旅行路线？*

# AI回答：
# 1. **地图服务与API整合：**
#    - 使用主流地图服务提供商的API，如Google Maps API、高德地图API或百度地图API。这些平台提供丰富的地理数据，支持动态交通信息的展示和路线规划。
#    - 结合用户行为数据和兴趣点推荐功能，通过地理信息可视化显示适合用户的路线和地点。

# 2. **多模式交通分析：**
#    - 借助地理数据可视化，展示多种出行方式（如驾车、步行、公共交通）的时间和成本对比。
#    - 提供动态切换功能，例如使用热力图显示交通流量，帮助用户避开拥堵。

# 3. **交互式地图设计：**
#    - 设计用户友好的交互界面，支持缩放、旋转和图层切换功能，让用户可以从宏观视角观察城市交通，也能查看微观细节如停车位分布。

# 4. **基于数据驱动的个性化推荐：**
#    - 结合用户历史行为数据，推荐用户可能感兴趣的旅游景点、酒店和餐饮，并根据用户选择的路线提供实时调整。
#    - 利用天气和实时活动信息，通过地图标注向用户提醒关键动态。

# 5. **数据分析支持与技术实现：**
#    - 提供算法支持，例如Dijkstra算法或A*算法，用于快速计算最短路径。
#    - 整合地理数据库（如PostGIS）和可视化库（如D3.js、Leaflet）构建系统。

# 6. **实际案例支持：**
#    - 引用现有的成功案例，例如Uber的动态路径规划和Citymapper的多模式交通解决方案，展示技术如何在实际场景中帮助用户。

# 通过以上详细回答，帮助用户全面理解如何运用地理数据可视化技术来优化旅行规划过程。
# """

FINANCE_PROMPT = """

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
