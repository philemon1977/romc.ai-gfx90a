from vllm import LLM, SamplingParams
llm = LLM(model="/models", max_model_len=512, enforce_eager=True,
          gpu_memory_utilization=0.2)
out = llm.generate(["Hello from ROCm on MI250, I am"],
                   SamplingParams(max_tokens=16, temperature=0))
print("COMPLETION:", repr(out[0].outputs[0].text))
print("VLLM: PASS")
