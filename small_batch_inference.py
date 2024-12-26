import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

# 加载模型和分词器
model_name = "gpt2"  # 可替换为更大的模型（如 EleutherAI/gpt-j-6B）
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DeepSpeed 推理初始化
ds_engine = deepspeed.init_inference(
    model=model,
    mp_size=1,  # 模型并行规模 (1 表示单 GPU)
    dtype=torch.float16,  # 使用 FP16 优化
    replace_with_kernel_inject=True,  # 替换优化内核
)

# 推理输入
prompt = "DeepSpeed is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 推理
with torch.no_grad():
    outputs = ds_engine.generate(inputs["input_ids"], max_length=50)
    print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
