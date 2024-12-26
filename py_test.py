import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "EleutherAI/gpt-j-6B"  # 替换为目标模型
model = AutoModelForCausalLM.from_pretrained(model_name).half().to("cuda").eval()  # 加载到 GPU 并设置为 eval 模式
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置测试输入
batch_size = 16  # 批次大小
sequence_length = 128  # 输入序列长度
max_new_tokens = 8  # 每个输入生成的新 token 数量

# 将 pad_token 设置为 eos_token
tokenizer.pad_token = tokenizer.eos_token

# 构造输入数据
instr = "PyTorch enables flexible inference."
while len(instr) < sequence_length:
    instr += " " + instr
prompts = [instr] * batch_size
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length).to("cuda")

# 热身推理（避免首次运行的加载开销影响性能）
with torch.no_grad():
    for _ in range(5):
        outputs = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)

# 测量推理吞吐量
start_time = time.time()
test_times = 10
for _ in range(test_times):
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)

torch.cuda.synchronize()  # 确保所有 GPU 操作完成
end_time = time.time()

# 统计生成的 token 数量
total_tokens = batch_size * max_new_tokens
elapsed_time = (end_time - start_time) / test_times
throughput = total_tokens / elapsed_time  # 每秒生成的 token 数量

# 打印结果
print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
print(f"Generated tokens per batch: {total_tokens}")
print(f"Elapsed time: {elapsed_time*1000:.2f} ms")
print(f"Throughput: {throughput:.2f} tokens/second")
