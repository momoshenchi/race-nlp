# from modelscope.msdatasets import MsDataset
# ds = MsDataset.load('BAAI/IndustryCorpus2_tourism_geography')

from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('BAAI/IndustryInstruction_Finance-Economics', subset_name='default', split='train')
# 打印数据集基本信息
print("数据集信息:")
print(ds)

# 获取第一条数据来查看格式
print("\n第一条数据示例:")
first_item = next(iter(ds))
print(first_item)

# 打印数据集的键
print("\n数据字段:")
print(list(first_item.keys()))
