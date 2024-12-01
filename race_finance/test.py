import json
import hashlib
import os
import traceback
import torch
from langchain.prompts import ChatPromptTemplate
import langchain
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
langchain.debug = True
from huggingface_hub import login
from langchain_experimental.tot.base import ToTChain   
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity
from typing import Tuple
from langchain_core.output_parsers.string import StrOutputParser

login() # You will be prompted for your HF key, which will then be saved locally
model_name = "Qwen/Qwen2.5-3B"
log_file = "run_log_qwen_tot.jsonl"  # 日志
ans_file = "model_logits_qwen_tot.jsonl"  # 推理结果
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# llm = HuggingFaceHub(repo_id=model_name, device=device)
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
    device=1,
    )
from langchain.chains import SequentialChain
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


class MyChecker(ToTChecker):     
    def evaluate(self, problem_description: str, thoughts: Tuple[str, ...] = ()) -> ThoughtValidity:
        return ThoughtValidity.VALID_FINAL
       
        
llm = ChatHuggingFace(llm=llm, verbose=True)
def generate_response(inp_list, unique_id_list, max_new_tokens=2048):
    try:
        input_prompts = []
        for inp in inp_list:
            messages = f"你的任务是回答与金融、经济相关的各种问题，包括选择题和开放式问题。对于选择题，请先直接给出正确答案，然后详细说明理由，并逐一分析其他选项的优缺点。你的回答应确保内容严谨、准确。\n问题如下: {inp}",
            template ="""
                Step 2:

                For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, initial effort needed, implementation difficulty, potential challenges, and the expected outcomes. Assign a probability of success and a confidence level to each option based on these factors

                {solutions}"""
          
            template3 ="""
            Step 4:

             Provide a justification for each ranking and offer any final thoughts or considerations for each solution
            {deepen_thought_process}

            A:  """
            
            

            # overall_chain = SequentialChain(
            #     chains=[chain1, chain2, chain3, chain4],
            #     input_variables=["input", "perfect_factors"],
            #     output_variables=["ranked_solutions"],
            #     verbose=True
            # )

            # messages = [msg_type(content=content) for msg_type, content in messages]
            # template = ChatPromptTemplate.from_messages(messages)
            # prompt0 = template.format()
            # print(prompt0)
            # prompt = "\n".join(message.content for message in prompt0)
            # print(prompt)
            
            input_prompts.append(messages)

            responses = []
            run_infos = []
            for i, prompt in enumerate(input_prompts):
                tot_chain = ToTChain(llm=llm, checker=MyChecker(), k=5, c=2, verbose=True, verbose_llm=False) 
                response =tot_chain.invoke(prompt)
                # response = llm.invoke(prompt)
                print(response)
                # if type(response) == str:
                responses.append(response['response'])
                run_info = {
                    "unique_id": unique_id_list[i],
                    "model_inp": prompt,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "top_p": 0.95,
                        "temperature": 0.8,
                        "do_sample": True
                    }
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