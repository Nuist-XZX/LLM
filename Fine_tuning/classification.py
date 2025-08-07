import sys
sys.path.append("C:/Users/ywp/Desktop/LLM") # 替换根目录路径

import urllib.request
import zipfile
import os
import time
import json
import torch
import tiktoken
import pandas as pd
import tensorflow as tf

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from GPT_Model.model import GPTModel, generate_text_simple
from pre_train.train import load_weights_into_gpt, text_to_token_ids, token_ids_to_text, download_and_load_gpt2
from pre_train.gpt_download import load_gpt2_params_from_tf_ckpt

# 微调分为分类微调和指令微调
# 分类微调

# *********************  第一阶段  **************************
# -------------------- 一、准备数据集 ------------------------
# 1. 下载、解压数据集
# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# zip_path = "sms_spam_collection.zip"
# extracted_path = "sms_spam_collection"
# data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# try:
#     download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
# except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
#     print(f"Primary URL failed: {e}. Trying backup URL...")
#     url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
#     download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path) 

# 2. 读取数据集，加载到pandas DataFrame中
data_file_path = r"C:/Users/ywp/Desktop/LLM/sms_spam_collection/SMSSpamCollection.tsv"
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

# -------------------- 四、添加分类头 ------------------------

# -------------------- 五、计算分类损失和准确率 ------------------------
# 计算分类准确率，利用softmax和argmax来计算所谓的分类准确率，它计算给定数据集中正确预测的百分比
# 定义分类准确率函数
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

# 定义分类损失函数--使用交叉熵损失
# `calc_loss_batch` 函数与第5章中的相同，不同之处在于我们只对优化最后一个标记 `model(input_batch)[:, -1, :]` 感兴趣，而不是所有标记 `model(input_batch)`
# 单批次
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# 定义损失函数--多批次
# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# *********************  第三阶段  **************************
# -------------------- 六、在有监督数据上微调模型 ------------------------
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # 禁用dropout，以产出稳定且可复现的结果
    with torch.no_grad(): # 禁用梯度跟踪，减少计算开销
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# 定义训练函数
# Overall the same as `train_model_simple` in chapter 5
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens 跟踪看到的训练样本数量 (`examples_seen`) 而不是看到的标记数量
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                    )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch 在每个epoch后计算准确率，而不是打印一个示例文本
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

# 绘制训练集和验证集损失函数以及准确率图表
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

# -------------------- 七、使用LLM作为垃圾消息分类器 ------------------------
# classify_review 类似于 SpamDataset
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )    
    # Alternatively, a more robust version is the following one, which handles the max_length=None case better
    # max_len = min(max_length,supported_context_length) if max_length else supported_context_length
    # input_ids = input_ids[:max_len]
    
    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"


if __name__=='__main__':
    # ###################################### 第一步：
    # 3. 创建一个平衡的数据集
    balanced_df = create_balanced_dataset(df)
    # print(balanced_df["Label"].value_counts())
    # 将字符串类别标签“ham”和“spam”转换为整数类别标签0和1，即将文本转换为词元ID，这里只有0和1两个词元ID
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # 4. 划分训练集、验证集和测试集
    # train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # 保存为CSV文件
    # train_df.to_csv("train.csv", index=None)
    # validation_df.to_csv("validation.csv", index=None)
    # test_df.to_csv("test.csv", index=None)

    # ###################################### 第二步：
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

    # ###################################### 第三步：
    # 2. 加载预训练权重和模型
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    # settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    tf_ckpt_path = tf.train.latest_checkpoint("C:/Users/ywp/Desktop/LLM/gpt2/124M")
    settings = json.load(open("C:/Users/ywp/Desktop/LLM/gpt2/124M/hparams.json", "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

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
    # print(token_ids_to_text(token_ids, tokenizer))

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
    # print(token_ids_to_text(token_ids, tokenizer))

    # ###################################### 第四步：
    # 原GPT_Model的最后线性输出层是将768个隐藏单元的输出映射到词表大小50257的输出空间，而分类任务需要将输出映射到两个类别（“ham”和“spam”）上。
    # print(model)
    # 1. 首先冻结模型, 即将所有层设置为不可训练
    for param in model.parameters():
        param.requires_grad = False

    # 2. 替换输出层model.out_head，直接将参数中的emb_dim替换为2就行
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    # 替换完成后，后续只需要训练这一个层就够了
    # 另外可以将最后一个Transformer块和层归一化设置为可训练
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # 测试
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    # print("Inputs:", inputs)
    # print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)
    with torch.no_grad():
        outputs = model(inputs)
    # print("Outputs:\n", outputs)
    # print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)
    # 因为我们目标是区分垃圾文件，所以只需要关注output的最后一个输出词元，并将其转换成类别标签预测
    """
    # Note: 为什么只对最后一个输出词元关注就可以区分垃圾信息。
        这是因为我们的掩码注意力只需要关注当前和之前的信息，
        而它的最后一个词元是唯一一个计算前面所有词元注意力分数的词元，
        即最后一个词元累计了最多的信息，唯一可以访问之前所有的词元，
        所以对于垃圾文件区分，只需要关注最后一个词元就可以
    """

    # ###################################### 第五步：
    # 使用softmax和argmax来预测区分垃圾信息
    probas = torch.softmax(outputs[:, -1, :], dim=-1) # 选最后一个输出词元
    label = torch.argmax(probas) # 最大值索引 0：非垃圾 1：垃圾
    # print("Class label:", label.item())

    # 计算不同数据集分类准确率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes
    torch.manual_seed(123) # For reproducibility due to the shuffling in the training data loader

    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    # print(f"Training accuracy: {train_accuracy*100:.2f}%")
    # print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    # print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # 测试
    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    # print(f"Training loss: {train_loss:.3f}")
    # print(f"Validation loss: {val_loss:.3f}")
    # print(f"Test loss: {test_loss:.3f}")

    # ###################################### 第六步：
    # 开始训练
    start_time = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1) # 优化器
    num_epochs = 5 # 训练轮数
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # loss
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
    # accuracy
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    # 就算整个数据集在训练、验证和测试集上的性能指标
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # ###################################### 第七步：
    # 测试
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    torch.save(model.state_dict(), "review_classifier.pth")
    model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict)