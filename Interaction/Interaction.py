import torch
from GPT_Model.model import GPTModel
from Fine_tuning.chapter07 import BASE_CONFIG, model_configs
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from your_model_module import GPTModel  # 导入您的模型定义

# 1. 加载模型
model_path = "gpt2-medium355M-sft.pth"  # 您的.pth文件
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load(model_path))
model.eval()

# 2. 加载分词器（使用与训练相同的）
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# 3. 交互函数（同上）


# 1. 加载模型和分词器
model_path = "gpt2-medium355M-sft"  # 替换为您的保存路径

# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

# 加载配置（需要您的原始配置）
from pre_train.train import BASE_CONFIG, model_configs  # 导入您的配置

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 创建模型实例
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
model.eval()

# 2. 定义生成函数
def generate_response(prompt, max_length=200, temperature=0.7, top_p=0.9):
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码并返回
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. 交互循环
def format_input(entry):
    """与训练代码相同的格式化函数"""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

print("===== 模型交互系统 =====")
print("输入格式: [指令] (可选: [输入])\n例如: '写一首诗' 或 '翻译句子 你好世界'\n输入 'exit' 退出\n")

while True:
    user_input = input("您: ")
    
    if user_input.lower() == 'exit':
        break
    
    # 解析用户输入
    if ' ' in user_input:
        instruction, input_text = user_input.split(' ', 1)
        entry = {"instruction": instruction, "input": input_text}
    else:
        entry = {"instruction": user_input, "input": ""}
    
    # 格式化提示
    formatted_prompt = format_input(entry)
    
    # 生成响应
    response = generate_response(formatted_prompt, max_length=300)
    
    # 提取响应部分
    response_start = response.find("### Response:") + len("### Response:")
    model_response = response[response_start:].strip()
    
    print(f"\n模型: {model_response}\n")