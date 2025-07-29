import torch

# ------------------ 一、简化自注意力机制 ------------------
# "You journey starts with one step" --> 对应的词元嵌入向量, 用于自注意力机制的输入
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 1. 利用点积计算查询词元(输入词元)与其他词元的注意力分数  点积越大, 两个元素之间的相似度和注意力分数越高
query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0]) # 初始化注意力分数张量, 用于存储每个词元与查询词元的注意力分数
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

# print(attn_scores_2)

# 2. 归一化注意力分数生成注意力权重, 使其和为1，最好的方式是使用softmax函数
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

# ------------------ 简化softmax函数 用于归一化注意力分数------------------
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())

####################################################
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# print("Attention weights:", attn_weights_2)
# print("Sum:", attn_weights_2.sum())

# 3. 归一化后的注意力权重与输入词元嵌入向量相乘并求和，得到加权后的上下文向量
query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

# print(context_vec_2)

# ------------------ TEST: 实现简化自注意力机制的完整实现 ------------------
# 计算所有上下文向量
# 1. 计算注意力分数
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

# print(attn_scores)

# for循环太慢，使用矩阵乘法来计算注意力分数
attn_scores = inputs @ inputs.T
# print(attn_scores)

# 2. 计算注意力权重
attn_weights = torch.softmax(attn_scores, dim=-1)
# print(attn_weights)

# 3. 计算上下文向量
all_context_vecs = attn_weights @ inputs
# print(all_context_vecs)

# ------------------ 二、带可训练权重的自注意力机制  又称“缩放点积注意力” ------------------
# 以第二个输入元素journey为例
# 1.定义
x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3 输入维度
d_out = 2 # the output embedding size, d=2 输出维度

# 2.初始化三个权重矩阵（查询权重、键权重、值权重）
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 3.计算查询向量、键向量和值向量
# 通过与对应的输入词元嵌入相乘获得
query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value

# print(query_2)
# 矩阵乘法得到所有输入词元的查询、键和值向量
keys = inputs @ W_key 
values = inputs @ W_value

# print("keys.shape:", keys.shape)
# print("values.shape:", values.shape)

# 4.计算注意力分数
# 不再同简化自注意力机制一样直接进行点积，而是通过各自权重矩阵变换后的查询向量和键向量进行计算，即输入词元的查询向量点积其他各词元的键向量
keys_2 = keys[1] # Python starts index at 0
attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22)

# 矩阵乘法推广到所有的注意力分数
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
# print(attn_scores_2)

# 5.归一化生成注意力权重, 不再使用简单的softmax,而是通过将注意力分数除以键向量的嵌入维度的平方根来进行缩放
# 缩放点积注意力. 对嵌入维度进行归一化是为了避免梯度过小
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)

# 6.通过对值向量进行加权求和计算上下文向量
context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)

# ------------------ TEST: 实现简化带权重的自注意力python类------------------
import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))

# 使用nn.Linear实现自注意力机制, 即将权重矩阵定义为线性层, 因为linear提供了优化的权重初始化方案
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))

# ------------------ 三、因果自注意力(掩码注意力)机制的实现 ------------------
# 因果注意力机制的思想是，对于输入序列中的每个元素，只关注其前面和当前的元素, 隐藏未来词元，而不是整个序列。
# 在因果注意力中，掩码对角线以上的注意力权重，确保在计算上下文向量时，LLM无法访问利用未来的词元
# 因果注意力实现方式:掩码对角线以上归一化后的注意力权重,用0掩码,然后对掩码后的矩阵的每一行重新归一化得到掩码注意力权重
# 1. 按之前方法计算注意力权重, 方便起见用之前的SelfAttention_v2类的查询权重矩阵和键权重矩阵
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

# 2. 创建掩码矩阵, 掩码矩阵的形状与注意力权重相同, 上三角部分为0, 下三角部分为1
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)

# 3. 掩码矩阵与注意力全在矩阵相乘,使对角线以上为0
masked_simple = attn_weights*mask_simple
# print(masked_simple)

# 4. 对掩码后的矩阵的每一行重新归一化,得到掩码注意力权重
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)

# ------------------ 使用更高效的方法计算掩码注意力权重softmax ------------------
# 1. 对角线以上-∞, softmax会将-∞视为0概率
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)

# 2. 归一化生成掩码注意力权重
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

# ------------------ 利用dropout掩码额外注意力权重 ------------------
# 减少模型对特定隐藏层单元的以来,避免过拟合,仅在训练时使用
# 一般使用dropout的两个时间点: 1. 计算注意力权重之后  2. 将权重应用于值向量之后
# 例: 使用50%的dropout率, 这意味着掩码一半的注意力权重, 应用到因果注意力掩码注意力权重后, 具体看P71
# 1. 创建一个全1矩阵, 一半的值被置0, 为了补偿减少的活跃元素, 矩阵中剩余值会按1/0.5=2进行放大
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
example = torch.ones(6, 6) # create a matrix of ones

# print(dropout(example))

# 2. 将dropout应用于掩码注意力权重
torch.manual_seed(123)
# print(dropout(attn_weights))

# ------------------ TEST: 实现简化因果注意力类------------------
# 把因果注意力和dropout应用到SelfAttention_v2类中,这个类将成为开发多头注意力的基础
# 1. 模拟批量输入
batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

# 2. 定义因果注意力CausalAttention类, 继承自SelfAttention_v2类
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method. 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

# ------------------ 四、多头注意力的实现 ------------------
# 并行多个因果注意力实现多头注意力
# 一个实现多头注意力的封装类
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2 # num_heads 是几个注意力头
)

context_vecs = mha(batch)

# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

# ------------------ 高效多头注意力类 ------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs  输出投影层,是合并多个头之后的操作
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
        # this will result in errors in the mask creation further below. 
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forwar

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf) # 注意力分数
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # 注意力权重

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) # 上下文向量,并转置
        # 以上为因果注意力实现
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # 重塑(展平)为(b, num_tokens, self.d_out),方便整合所有头的输出
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)