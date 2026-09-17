#!/usr/bin/env python3
"""8107 (Qwen3.8-Flash-Next BF16) 的单流探针：就绪等待 + 预热丢弃 + n 次计时。
用法：qwen38_probe.py PORT MODEL [MAXTOK=64] [REPS=3] [single]
  single = 只发一次（给 rocprofv3 的采集窗口用）
口径纪律：固定 prompt + **丢弃首个请求** + 中位数（本会话教训：仅报 t/s 会把接受率变化误读成内核变快）。
"""
import json
import statistics
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8107"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
SINGLE = len(sys.argv) > 5 and sys.argv[5] == "single"

# 与 Ornith 臂同一条 count prompt：可预测性高、MTP 接受率稳定 ⇒ A/B 信噪比最好
PROMPT = "Count slowly from one to ninety, writing each number in words on its own line."


def one() -> tuple[float, int]:
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "max_tokens": MAXTOK,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    return dt, d["usage"]["completion_tokens"]


def wait_ready(timeout_s: int = 2400) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).read()
            return time.time() - t0
        except Exception:
            time.sleep(5)
    raise SystemExit("❌ 服务未就绪")


if __name__ == "__main__":
    print(f"[8107] 等待就绪 …", flush=True)
    print(f"[8107] ✅ ready（等 {wait_ready():.0f}s）", flush=True)
    if SINGLE:
        dt, n = one()
        print(f"[8107] 单发：{n} tok / {dt:.2f}s = {n/dt:.2f} tok/s", flush=True)
        sys.exit(0)
    dt, n = one()
    print(f"[8107] warmup(丢弃) {n} tok / {dt:.2f}s = {n/dt:.2f} tok/s", flush=True)
    ts = []
    for i in range(REPS):
        dt, n = one()
        ts.append(n / dt)
        print(f"[8107] rep{i+1}: {n} tok / {dt:.2f}s = {n/dt:.2f} tok/s", flush=True)
    print(f"[8107] MEDIAN {statistics.median(ts):.2f} tok/s  min {min(ts):.2f} max {max(ts):.2f} "
          f"spread {100*(max(ts)-min(ts))/statistics.median(ts):.1f}% (n={REPS}, 首个已丢)", flush=True)
