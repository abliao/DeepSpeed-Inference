import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

# 加载模型和分词器
model_name = "EleutherAI/gpt-neo-2.7B"  # 替换为更大的模型
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_gpus = 4

# DeepSpeed 推理初始化
ds_engine = deepspeed.init_inference(
    model=model,
    mp_size=num_gpus,  # 使用 4 个 GPU 进行张量并行
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)

# 推理输入
prompts = ["DeepSpeed is amazing"] * num_gpus  # 多批次推理
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

# 推理
with torch.no_grad():
    outputs = ds_engine.generate(inputs["input_ids"], max_length=50)
    for i, output in enumerate(outputs):
        print(f"Prompt {i}: {tokenizer.decode(output, skip_special_tokens=True)}")
