from huggingface_hub import login
import transformers
import torch

login(token="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")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)

print(pipeline("Hey how are you doing today?"))

