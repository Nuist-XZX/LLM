import sys
sys.path.append("C:/Users/ywp/Desktop/LLM") # 替换根目录路径

import urllib.request
import zipfile
import os
import torch
import tiktoken
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from GPT_Model.model import GPTModel, generate_text_simple
from pre_train.train import load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from pre_train.gpt_download import download_and_load_gpt2

# 微调分为分类微调和指令微调
# 分类微调

# *********************  第一阶段  **************************
# -------------------- 一、准备数据集 ------------------------
# 1. 下载、解压数据集
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

# def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
#     if data_file_path.exists():
#         print(f"{data_file_path} already exists. Skipping download and extraction.")
#         return

#     # Downloading the file
#     with urllib.request.urlopen(url) as response:
#         with open(zip_path, "wb") as out_file:
#             out_file.write(response.read())

#     # Unzipping the file
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extracted_path)

#     # Add .tsv file extension
#     original_file_path = Path(extracted_path) / "SMSSpamCollection"
#     os.rename(original_file_path, data_file_path)
#     print(f"File downloaded and saved as {data_file_path}")

# try:
#     download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
# except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
#     print(f"Primary URL failed: {e}. Trying backup URL...")
#     url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
#     download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path) 

# 2. 读取数据集，加载到pandas DataFrame中
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
# print(df)

# 3. 创建一个平衡的数据集
def create_balanced_dataset(df):
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())
# 将字符串类别标签“ham”和“spam”转换为整数类别标签0和1，即将文本转换为词元ID，这里只有0和1两个词元ID
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# 4. 划分训练集、验证集和测试集，并保存为CSV文件
# 这里使用随机划分法，将数据集划分为训练集、验证集和测试集，训练集占70%，验证集占10%，测试集占20%
def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

# train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# 保存为CSV文件
# train_df.to_csv("train.csv", index=None)
# validation_df.to_csv("validation.csv", index=None)
# test_df.to_csv("test.csv", index=None)

# -------------------- 二、创建数据加载器 ------------------------
# 对文本长度不够的文本利用填充次元 `<|endoftext|>` 的词元ID50256进行填充
# 1. 首先实现一个Dataset类，之后再实例化数据加载器
class SpamDataset(Dataset):
    """
    将文本转换为词元ID, 识别数据集中最长的序列,
    将文本长度不够的文本使用填充词元50256进行填充
    """
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

# 创建分词器
tokenizer = tiktoken.get_encoding("gpt2")
# 从train.csv加载数据集
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
# print(train_dataset.max_length) # 120
# 验证集
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
# 测试集
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# 2. 实例化数据加载器
num_workers = 0
batch_size = 8 # 批次大小，每个批次有长度为120的词元
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# 打印查看
# print("Train loader:")
# for input_batch, target_batch in train_loader:
#     pass
# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions", target_batch.shape)
# # 每个数据集中的总批次数量
# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")

# *********************  第二阶段  **************************
# -------------------- 三、初始化带有预训练权重的模型 ------------------------
# 1. 配置参数
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 2. 加载预训练权重和模型
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# 验证模型权重是否成功加载到GPTModel
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

# 测试模型能否分类
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

