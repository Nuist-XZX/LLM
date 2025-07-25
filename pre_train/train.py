import sys
sys.path.append("C:/Users/ywp/Desktop/LLM")

import torch
import numpy as np
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from GPT_Model.model import GPTModel, generate_text_simple
from dataloader.dataset import create_dataloader_v1
from pre_train.gpt_download import download_and_load_gpt2

# ---------------------- 一、评估文本生成模型 --------------------------
# 1. 文本生成
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024) 降低到256减少计算资源需求
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference

# 分词器
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

# 解码器
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
    )

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# ------------------------------------------------------------------------------ #
# 假设两个文本
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

# 推理计算每个词元的logits并转成概率
with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
# print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)

# 从最后一个维度选择概率最高的那个索引作为预测词元的ID
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)

# print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# ------------------------------------------------------------------------------ #

# 2. 计算文本生成损失 -- 交叉熵和困惑度
# 第一步、第二步分别为模型输出logits和softmax得到概率
# 第三步：得到目标词元的概率
# 打印与目标词元对应的初始softmax概率分数
text_idx = 0 # 第一个文本
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 1:", target_probas_1)
text_idx = 1 # 第二个文本
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 2:", target_probas_2)

# 第四步：计算对数概率
# Compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# print(log_probas)

# 第五步：计算平均对数概率
# Calculate the average probability for each token
avg_log_probas = torch.mean(log_probas)
# print(avg_log_probas)

# 第六步：计算负平均对数概率，即交叉熵损失
neg_avg_log_probas = avg_log_probas * -1
# print(neg_avg_log_probas)

# 使用pytorch的cross_entropy函数执行前面的步骤，计算交叉熵损失
# 先检查一下logits和targets的形状
# Logits have shape (batch_size, num_tokens, vocab_size)
# print("Logits shape:", logits.shape)
# Targets have shape (batch_size, num_tokens)
# print("Targets shape:", targets.shape)
# 展平张量
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)
# 计算交叉熵损失
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print(loss)
# 计算困惑度，困惑度越低，预测更接近实际分布。困惑度通常与交叉熵损失一起用来评估模型的性能。
# - 困惑度通常被认为更具可解释性，因为它可以理解为模型在每一步不确定的有效词汇表大小（如：perplexity=48,725个单词或标记）。
# - 换句话说，困惑度提供了一种衡量模型预测的概率分布与数据集中实际单词分布匹配程度的方法。
# - 类似于损失，较低的困惑度表示模型预测更接近实际分布。
perplexity = torch.exp(loss)
# print(perplexity) # tensor(48725.8203)

# 3. 计算训练集和验证集的损失，将 2 中的损失计算应用到文本数据集中
# 使用短篇小说 The Verdict 作为数据集
# import os
# import urllib.request
# 保存短篇小说
# if not os.path.exists("the-verdict.txt"):
#     url = ("https://raw.githubusercontent.com/rasbt/"
#            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#            "the-verdict.txt")
#     file_path = "the-verdict.txt"
#     urllib.request.urlretrieve(url, file_path)

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
# 检查一下数据集中的字符数和词元数
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:", total_tokens)

# 90%训练、10%验证
# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 使用creat_dataloader_v1函数创建训练集和验证集的dataloader
torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# 检查
# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

# 定义计算单批次的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
# 定义计算多批次的损失
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
# 将calc_loss_loader函数应用到训练集和验证集的dataloader上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes
with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
# print("Training loss:", train_loss) # Training loss: 10.987583584255642
# print("Validation loss:", val_loss) # Validation loss: 10.98110580444336

# ---------------------- 二、训练大语言模型 --------------------------
# 1. 定义训练大模型的主函数
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 重置上一个批次迭代中的损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 计算损失相对于模型参数的梯度。这是通过自动微分实现的，loss.backward()会填充模型参数的.grad属性。
            optimizer.step() # 使用优化器更新模型参数。optimizer.step()会应用优化算法（例如，Adam、SGD等）来调整模型的权重和偏差，以最小化损失。
            tokens_seen += input_batch.numel() # 跟踪训练集中已处理的词元数
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter) # 打印训练集和验证集损失
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每轮之后打印一个文本样本
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # 禁用dropout，以产出稳定且可复现的结果
    with torch.no_grad(): # 禁用梯度跟踪，减少计算开销
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

# 2. 训练LLM
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# 3. 创建画布，把训练和验证损失做成曲线图
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    # fig.tight_layout()  # Adjust layout to make room
    # plt.savefig("loss-plot.pdf")
    # plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# ---------------------- 三、文本生成（解码）策略 --------------------------
# 控制随机性的解码策略，以生成更具原创性的文本
# 通过使用温度缩放和Top-K采样来改进generate_text_simple函数，实现更具随机性的文本生成
model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25, # 最多生成多少个新的词元tokens
    context_size=GPT_CONFIG_124M["context_length"]
)

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# 1. 温度缩放策略：用一个概率分布中采样的函数来取代之前的torch.argmax()函数，以生成更具原创性的文本。
# 示例：概率采样
vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
}  # 词汇表
inverse_vocab = {v: k for k, v in vocab.items()}

# 假设输入上下文为 "every effort moves you"，并生成下一个词元的logits
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
# print(inverse_vocab[next_token_id])

# 为了使用概率分布，使用multinomial函数替换argmax
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
# print(inverse_vocab[next_token_id])

# 重复采样1000次，看看会发生什么
def print_sampled_tokens(probas):
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)
# ------------------ 以上为概率分布概念 ----------------------

# 接下来使用温度缩放概念来控制分布和选择过程。
# 温度缩放是指将logits除以一个大于0的数，温度大于1会在应用softmax后导致标记概率分布更加均匀，温度小于1会在应用softmax后导致标记概率分布更加集中（更尖锐或更峰值）。
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence
# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()

plt.tight_layout()
plt.savefig("temperature-plot.pdf")
plt.show()

# 2. Top-K采样
# 为了能够使用更高的温度增加输出的多样性并减少生成无意义句子的概率，将采样的标记限制在最有可能的top-k个标记中。
# Top-K将采样的词元限制在前K个最可能的词元上，并通过掩码概率分数的方式来排除其他词元。
# Top-k用负无穷值（-inf）替换所有未选择的logits，在计算softmax时，非前k词元的概率分数为0，剩余概率总和为1.
# 示例
# 先选择logits最高的前3个词元logits
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
# print("Top logits:", top_logits)
# print("Top positions:", top_pos)

# 然后利用where将低于前3个词元中最低logits值的词元的logits设置为负无穷（-inf）
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")), 
    other=next_token_logits
)
# 更高效的写法
# new_logits = torch.full_like( # create tensor containing -inf values
#     next_token_logits, -torch.inf
# )   
# new_logits[top_pos] = next_token_logits[top_pos] # copy top k values into the -inf tensor

# print(new_logits) # tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])

# 最后应用softmax将logits转换为下一个词元的概率
topk_probas = torch.softmax(new_logits, dim=0)

# 3. 修改文本生成函数generate_text_simple，创建一个新的generate函数，将温度缩放和Top-K应用在函数中
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

# 测试generate
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# ---------------------- 四、加载和保存权重 --------------------------
torch.save(model.state_dict(), "model.pth") # 保存模型权重

# 加载模型权重
model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

# 保存模型和优化器状态
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer.pth"
)

# 加载模型和优化器状态
checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
# ---------------------- 五、从OpenAI加载预训练权重 --------------------------
# 下载权重
# 加载GPT-2架构设置settings和权重参数params
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# 将权重转移到GPTModel
# 1. 初始化GPTModel实例
# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
# 请注意，原始的GPT模型在多头注意力模块中初始化查询、键和值矩阵的线性层时使用了偏置向量，这在我们的情况下不是必需或推荐的；然而，为了能够正确加载权重，我们还需要在实现中启用这些偏置向量，即将`qkv_bias`设置为`True`。
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

# 2. 将OpenAI的权重分配给GPTModel实例中对应的权重张量
# 定义一个工具函数，检查两个张量或数组是否具有相同的维度或形状，并将right张量返回可训练的pytorch参数。
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# 定义权重加载函数
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
# 应用load_weights_into_gpt函数加载权重
load_weights_into_gpt(gpt, params)
gpt.to(device)

# 模型加载成功后，使用generate生成新文本
torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))