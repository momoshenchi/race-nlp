import json
from modelscope.msdatasets import MsDataset

# 通过deita_score过滤sft数据
# 加载数据集
ds =  MsDataset.load('BAAI/IndustryInstruction_Travel-Geography', subset_name='default', split='train')
print("数据集信息:")
print(ds)
# 过滤掉 deita_score 小于 8 的项
filtered_data = [item for item in ds if item['deita_score'] >= 8]

# 提取并组织 conversations 字段
conversations_data = []
for item in filtered_data:
    conversations_data.append({"conversations": item['conversations']})

# 将结果保存为 JSON 文件
with open('conversations.json', 'w', encoding='utf-8') as f:
    json.dump(conversations_data, f, ensure_ascii=False, indent=4)

# 输出记录的个数
print(f"Total records saved: {len(conversations_data)}")
