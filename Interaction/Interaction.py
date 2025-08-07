import sys
sys.path.append("C:/Users/ywp/Desktop/LLM") # 替换根目录路径

import torch
import tiktoken

from GPT_Model.model import GPTModel
from Fine_tuning.personal import format_input
from pre_train.train import *

# 配置文件
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

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
context_size = BASE_CONFIG["context_length"]  # 使用配置中的上下文长度

# 1. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:/Users/ywp/Desktop/LLM/gpt2-medium355M-sft.pth"  # 您的.pth文件
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 2. 加载分词器（使用与训练相同的）
tokenizer = tiktoken.get_encoding("gpt2")
eot_token_id = tokenizer.eot_token  # 获取结束token ID 即50256

# 3. 交互循环
print("===== 模型交互系统 =====")
print("输入格式: [指令] (可选: [输入])")
print("例如: '写一首诗' 或 '翻译句子 你好世界'")
print("输入 'exit' 退出\n")

while True:
    user_input = input("您: ")
    
    if user_input.lower() == 'exit':
        break
    
    # 解析用户输入
    if ' ' in user_input:
        parts = user_input.split(' ', 1)
        instruction = parts[0]
        input_text = parts[1] if len(parts) > 1 else ""
        entry = {"instruction": instruction, "input": input_text}
    else:
        entry = {"instruction": user_input, "input": ""}
    
    # 格式化提示
    formatted_prompt = format_input(entry)
    print(f"\n完整提示: {formatted_prompt}")
    
    # 将文本转换为token IDs
    input_ids = text_to_token_ids(formatted_prompt, tokenizer)
    input_tensor = input_ids.to(device)
    
    # 生成响应 - 使用您的generate函数
    try:
        output_ids = generate(
            model=model,
            idx=input_tensor,
            max_new_tokens=300,
            context_size=context_size,
            temperature=1.4,  # 控制多样性
            top_k=50,         # 限制候选词范围
            eos_id=eot_token_id  # 结束token ID
        )
        
        # 将token IDs转换回文本
        full_response = token_ids_to_text(output_ids[0], tokenizer)
        
        # 提取模型响应部分
        response_start = full_response.find("### Response:")
        if response_start != -1:
            model_response = full_response[response_start + len("### Response:"):].strip()
        else:
            model_response = full_response[len(formatted_prompt):].strip()
        
        print(f"\n模型: {model_response}\n")
    
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("\n内存不足! 请尝试缩短输入或使用更小的批次大小。\n")
        else:
            print(f"\n生成时出错: {e}\n")