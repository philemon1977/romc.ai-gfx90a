import time


def main():
    t0 = time.perf_counter()
    from vllm import LLM, SamplingParams

    MODEL = "/mnt/stripe-3mix-3t2/models/Qwen/Qwen3.8-27B"
    print(f"[import] {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    llm = LLM(model=MODEL, tensor_parallel_size=2, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.90,
              trust_remote_code=True)
    load_s = time.perf_counter() - t0
    print(f"[load+engine-init] {load_s:.1f}s", flush=True)

    # ---- A. 单流正确性 + TPS ----
    prompts_a = [
        "Explain in two sentences what a mixture-of-experts layer is.",
        "请用一句话介绍ROCm软件栈。",
    ]
    sp_a = SamplingParams(temperature=0, max_tokens=128)
    t0 = time.perf_counter()
    outs = llm.generate(prompts_a, sp_a)
    dt = time.perf_counter() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\n=== A. single: {n_tok} tok in {dt:.1f}s -> {n_tok/dt:.1f} tok/s", flush=True)
    for o in outs:
        print(f"  Q: {o.prompt[:60]!r}\n  A: {o.outputs[0].text.strip()[:220]!r}", flush=True)

    # ---- B. 8 并发吞吐 ----
    prompts_b = [f"Count from 1 to 60, one number per line. Start now: {i}" for i in range(8)]
    sp_b = SamplingParams(temperature=0, max_tokens=256)
    t0 = time.perf_counter()
    outs_b = llm.generate(prompts_b, sp_b)
    dt_b = time.perf_counter() - t0
    n_b = sum(len(o.outputs[0].token_ids) for o in outs_b)
    print(f"\n=== B. conc8: {n_b} tok in {dt_b:.1f}s -> {n_b/dt_b:.1f} tok/s aggregate "
          f"({n_b/dt_b/8:.1f} tok/s/req)", flush=True)
    print("B sample out:", repr(outs_b[0].outputs[0].text[:120]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
