import sys
sys.path.append("C:/Users/ywp/Desktop/LLM") # 替换根目录路径

import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from self_attention.attention import MultiHeadAttention

# 定义一个小型的GPT-2模型配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size 词元ID大小
    "context_length": 1024, # Context length 输入序列长度
    "emb_dim": 768,         # Embedding dimension 词嵌入维度
    "n_heads": 12,          # Number of attention heads 注意力头数
    "n_layers": 12,         # Number of layers Transformer层数
    "drop_rate": 0.1,       # Dropout rate dropout率
    "qkv_bias": False       # Query-Key-Value bias 
}

# 一个简化的GPT类模型实现
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 使用占位符代替TransformerBlock
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # 使用占位符代替LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds              # 词元嵌入和位置嵌入相加
        x = self.drop_emb(x)                     # 应用dropout
        x = self.trf_blocks(x)                   # 应用Transformer块
        x = self.final_norm(x)                   # 层归一化，先使用占位符DummyLayerNorm代替
        logits = self.out_head(x)                # 线性输出生成logits
        return logits

# DummyTransformerBlock和DummyLayerNorm是一个简单的占位符类，代替DummyGPTModel中的Transformer块，后续会被替换成真正的Transformer块
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.
    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x

# Note: 在大语言模型中，输入词元嵌入维度与输出维度相匹配。

# ------------------- 下面是利用DummyGPTModel实例化的GPT主干 -------------------
# 1. 使用tiktoken对包含两个文本输入的批次进行分词处理
tokenizer = tiktoken.get_encoding("gpt2") # 使用GPT-2的分词器

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0) # 词元ID序列
# print(batch) 

# 2. 实例化一个DummyGPTModel类，将分词后的数据传给它
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)

# ------------------- 开始构建model中其他的Transformer块 -------------------

# 1. 构建层归一化进行归一化激活
# 层归一化主要思想：调整时间网络层的输出，使其均值为0且方差为1。多应用于多头注意力模块前后和最终输出层之前
# 封装一个曾归一化的类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                                    # 加上方差，防止除以0错误
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                # 计算最后一个维度（嵌入维度emb_dim）的均值，keepdim=True确保输入张量和输出张量保持一样的维度
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算最后一个维度的方差，unbiased=False表示使用样本数量作为方差的除数，没有使用贝塞尔修正。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)   # 层归一化 = (层输出 - 均值) / ((方差 + self.eps)的平方根)
        return self.scale * norm_x + self.shift            # scale和shift为可训练参数，在训练过程中会自动调整这些参数用来找到最佳的缩放和偏移，提高模型的性能

# 测试LayerNorm
torch.manual_seed(123)
batch_example = torch.randn(2, 5)   # 创建一个示例批次，2个样本，每个样本5维
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# 2. 构建具有GRLU激活函数的前馈神经网络
# GPT-2 GELU激活函数的实现
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
# 绘制relu和gelu的图像
gelu, relu = GELU(), nn.ReLU()
# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
# plt.show()

# 使用GELU构建小型前馈神经网络模块类
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# 测试FeedForward
ffn = FeedForward(GPT_CONFIG_124M)
# input shape: [batch_size, num_token, emb_size]
x = torch.rand(2, 3, 768) 
out = ffn(x)
# print(out.shape)

# 3. 构建跳跃连接，缓解梯度消失
# 示例：实现一个5层深度神经网络，每层由一个线性层和一个GELU组成
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # 计算当前层的输出
            layer_output = layer(x)
            # 检查是否可以应用快捷连接
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
    
# 打印梯度值
def print_gradients(model, x):
    # 前向传播
    output = model(x)
    target = torch.tensor([[0.]])
    # 根据目标和输出的接近程度计算损失
    loss = nn.MSELoss()
    loss = loss(output, target)
    # 反向传播以计算梯度
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 印权重的平均绝对梯度
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# 测试ExampleDeepNeuralNetwork
# use_shortcut切换跳跃连接true和非跳跃连接false
layer_sizes = [3, 3, 3, 3, 3, 1]  
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
# print_gradients(model_without_shortcut, sample_input)

# 4. 将掩码多头注意力、层归一化、GELU、前馈神经网络和跳跃连接集成到一个Transformer块中
# GPT的Transformer块组件
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块的跳跃连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 加回原始输入

        # 前馈网络块的跳跃连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 加回原始输入

        return x

# 测试TransformerBlock
torch.manual_seed(123)
x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

# print("Input shape:", x.shape)
# print("Output shape:", output.shape)

# ------------------- 构建完整的GPT model -------------------
# 通过重复Trnasformer块构建GPT模型
# GPT模型架构
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# 测试GPTModel
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)                  # batch 是文本转换后的词元ID序列
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

# 打印模型参数总量 GPT-2模型参数为124M，但是这边打印为163M，是因为词元嵌入层作为输出层重复使用，依次要减去输出层参数量
total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")

total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

# 打印GPTModel的内存需求
# Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4
# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)
# print(f"Total size of the model: {total_size_mb:.2f} MB")

# ------------------- 将GPT张量输出转换为文本 -------------------
# 生成文本函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)  n_tokens指每个样本中词元（tokens）的数量
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx # 返回的是词元ID序列，后面再用分词器的decode函数转换为文本

# 示例：生成文本
# 首先生成输入文本的词元ID序列
start_context = "The mountains and rivers are shrouded in mist and rain,"   # 江山雾笼烟雨遥，十年一剑斩皇朝
encoded = tokenizer.encode(start_context)            # 转换成词元ID序列
# print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加batch维度
# print("encoded_tensor.shape:", encoded_tensor.shape)

# 然后调用generate_text_simple函数得到输出词元ID序列
model.eval() # disable dropout
out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=GPT_CONFIG_124M["context_length"]
)
# print("Output:", out)
# print("Output length:", len(out[0]))

# 最后使用分词器的.decode将词元ID序列转换为文本
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text) # The mountains and rivers are shrouded in mist and rain, kinderg Kir Gelcca Mons marched