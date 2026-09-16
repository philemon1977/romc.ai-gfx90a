import time


def main():
    from vllm import LLM, SamplingParams

    MODEL = "/mnt/stripe-3mix-3t2/models/Qwen/Qwen3.8-27B"
    t0 = time.perf_counter()
    llm = LLM(model=MODEL, tensor_parallel_size=2, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.90,
              trust_remote_code=True)
    print(f"[engine-init] {time.perf_counter()-t0:.1f}s", flush=True)

    # 预热（消除首个 decode 批的抖动）
    llm.generate(["warmup"], SamplingParams(max_tokens=1))

    print(f"\n{'conc':>5} {'out_tok':>8} {'sec':>7} {'agg tok/s':>10} {'tok/s/req':>10}", flush=True)
    for conc in (1, 8, 16, 32, 64):
        prompts = [f"Write a short paragraph about number {i}." for i in range(conc)]
        sp = SamplingParams(temperature=0, max_tokens=128)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"{conc:>5} {n:>8} {dt:>7.1f} {n/dt:>10.1f} {n/dt/conc:>10.1f}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
