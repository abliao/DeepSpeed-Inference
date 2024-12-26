import time
from transformers import AutoTokenizer
from fastertransformer import GPTInference

# 加载 FasterTransformer 模型
model_name = "gpt2"  # 替换为目标模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载 GPT 模型到 FasterTransformer（需事先转换模型）
ft_model = GPTInference(
    model_name=model_name,
    tensor_parallel_size=1,  # 单 GPU 推理
    data_type="fp16",  # 使用 FP16 优化
    model_path="./fastertransformer_models/gpt2"  # 替换为已转换的 FasterTransformer 模型路径
)

# 配置测试输入
batch_size = 8
sequence_length = 128
max_new_tokens = 8
tokenizer.pad_token = tokenizer.eos_token

instr = "FasterTransformer enables high-performance inference."
while len(instr) < sequence_length:
    instr += " " + instr
prompts = [instr] * batch_size
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)

# 热身推理
for _ in range(5):
    outputs = ft_model.generate(
        inputs=inputs["input_ids"],
        max_length=sequence_length + max_new_tokens
    )

# 测量推理吞吐量
start_time = time.time()
test_times = 10
for _ in range(test_times):
    outputs = ft_model.generate(
        inputs=inputs["input_ids"],
        max_length=sequence_length + max_new_tokens
    )

end_time = time.time()

# 统计生成的 token 数量
total_tokens = max_new_tokens
elapsed_time = (end_time - start_time) / test_times
throughput = total_tokens / elapsed_time

# 打印结果
print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
print(f"Generated tokens per batch: {total_tokens}")
print(f"Elapsed time: {elapsed_time:.5f} seconds")
print(f"Throughput: {throughput:.2f} tokens/second")
