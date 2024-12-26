
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

model_name = "EleutherAI/gpt-j-6B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, revision="float32", torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

deepspeed_config = {
        "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
        },
    }
}

model = deepspeed.init_inference(
    model,
    args=deepspeed_config,
    dtype=model.dtype,
    replace_method="auto",
    replace_with_kernel_inject=True,
)