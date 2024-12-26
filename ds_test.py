import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import deepspeed
from transformers import pipeline

# 加载模型和 DeepSpeed 初始化
model_name = "EleutherAI/gpt-neo-2.7B"  # 替换为目标模型 EleutherAI/gpt-neo-2.7B  gpt-j-6B gpt-neox-20b
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

config ={
        # "zero_optimization": {
        # "stage": 3,
        # "offload_param": {
        #     "device": "cpu"
        # },
        # },
        "quantization": {
            "enabled": True,
            "dtype": "int8",
            "opt_level": "O3"
        }
}
# 使用 DeepSpeed 初始化推理引擎
ds_engine = deepspeed.init_inference(
    model=model,
    args=config,
    tensor_parallel={"tp_size": 1},  # 单 GPU 或指定张量并行大小
    dtype=torch.float32,  # 使用 FP16 优化
    replace_with_kernel_inject=True,
)


# 配置测试输入
batch_size = 16  # 批次大小
sequence_length = 128  # 输入序列长度
max_new_tokens = 8  # 每个输入生成的新 token 数量

# 将 pad_token 设置为 eos_token
tokenizer.pad_token = tokenizer.eos_token

# 构造输入数据
instr = "DeepSpeed enables efficient inference."
while len(instr)<sequence_length:
    instr+=" "+instr
prompts = [instr] * batch_size
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,max_length=sequence_length).to("cuda")

# 测量推理吞吐量

for _ in range(5):
    outputs = ds_engine.generate(inputs["input_ids"], max_new_tokens=max_new_tokens, use_cache=False)

start_time = time.time()

test_times = 10
for i in range(test_times):
    with torch.no_grad():
        outputs = ds_engine.generate(inputs["input_ids"], max_new_tokens=max_new_tokens, use_cache=False)

torch.cuda.synchronize()
end_time = time.time()

# 统计生成的 token 数量
total_tokens = batch_size * max_new_tokens
elapsed_time = (end_time - start_time)/test_times
throughput = total_tokens / elapsed_time  # 每秒生成的 token 数量

# 打印结果
print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
print(f"Generated tokens per batch: {total_tokens}")
print(f"Elapsed time: {elapsed_time*1000:.2f} ms")
print(f"Throughput: {throughput:.2f} tokens/second")
