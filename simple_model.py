import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
import json
from torch.nn import functional as F

# ======================
# 数据加载模块
# ======================
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        
        # 读取并预处理数据
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        
        # 创建训练样本
        for i in range(0, len(tokens) - block_size, block_size):
            self.data.append(tokens[i:i+block_size+1])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class Tokenizer:
    def __init__(self, vocab_file=None):
        if vocab_file:
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
        else:
            # 创建基础字符级tokenizer
            chars = sorted(list(set(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")))
            self.vocab = {ch: i for i, ch in enumerate(chars)}
        
        self.itos = {i: ch for ch, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text):
        return [self.vocab[ch] for ch in text if ch in self.vocab]
    
    def decode(self, tokens):
        return ''.join([self.itos.get(token, '') for token in tokens])

# ======================
# 模型定义
# ======================
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 分割嵌入维度到多个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # 注意力计算
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        # 残差连接和层归一化
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        out = self.fc_out(out)
        return out

class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        embed_size=768,
        num_layers=12,
        heads=12,
        forward_expansion=4,
        dropout=0.1,
        max_length=1024
    ):
        super(GPT2, self).__init__()
        self.decoder = Decoder(
            vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        )
        self.max_length = max_length
    
    def make_mask(self, target):
        N, seq_length = target.shape
        mask = torch.tril(torch.ones((seq_length, seq_length))).expand(N, 1, seq_length, seq_length)
        return mask.to(target.device)
    
    def forward(self, x):
        mask = self.make_mask(x)
        return self.decoder(x, mask)
    
    def generate(self, context, max_len=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            generated = context
            
            for _ in range(max_len):
                # 截断超过最大长度的输入
                inputs = generated[:, -self.max_length:]
                mask = self.make_mask(inputs)
                
                # 预测下一个token
                outputs = self.decoder(inputs, mask)
                logits = outputs[:, -1, :] / temperature
                probabilities = F.softmax(logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probabilities, 1)
                generated = torch.cat((generated, next_token), dim=1)
            
            return generated

# ======================
# 训练模块
# ======================
def train_model(config):
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    
    # 创建数据集
    train_dataset = TextDataset(config['data_path'], tokenizer, config['block_size'])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # 初始化模型
    model = GPT2(
        vocab_size=tokenizer.vocab_size,
        embed_size=config['embed_size'],
        num_layers=config['num_layers'],
        heads=config['num_heads'],
        max_length=config['block_size']
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch+1) % config['save_interval'] == 0:
            torch.save(model.state_dict(), f"gpt2_epoch_{epoch+1}.pth")
    
    # 保存最终模型
    torch.save(model.state_dict(), "gpt2_final.pth")
    print("训练完成！")

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    config = {
        'data_path': 'input.txt',      # 训练数据文件
        'batch_size': 16,              # 批量大小
        'block_size': 256,             # 文本块大小
        'embed_size': 512,             # 嵌入维度
        'num_layers': 6,               # Transformer层数
        'num_heads': 8,                # 注意力头数
        'learning_rate': 3e-4,         # 学习率
        'num_epochs': 10,              # 训练轮数
        'save_interval': 5             # 保存间隔(epoch)
    }
    
    train_model(config)