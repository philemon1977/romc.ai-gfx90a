#!/usr/bin/env bash
# The aiter arm of the int8 A/B is *already running*: its gfx90a JIT build of
# module_gemm_a8w8 (one-time, ~20-40 min, serialized across 8 ranks on the baton lock)
# was still going when ab_int8.sh's 1000 s health poll gave up — and that script's
# NOT-READY path does not kill the server, so the instance survived and is building.
#
# So: wait for it, measure it, stop it, then chain the MTP-depth arms
# (mtp_depth.sh waits for the GPU itself, so this stays strictly serialized).
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
PIDF=/home/qiba/ai/logs/ornith397b-8115.pid
LOG="$REPO/logs/aiter_finish.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== aiter_finish start $(date -u +%FT%TZ) ==="

SRV=/home/qiba/ai/logs/ornith397b/server-8115-20260917-0536.log
ok=0
for i in $(seq 1 360); do   # up to 60 min: JIT build + 286 s weight load
  if curl -sf http://127.0.0.1:8115/health >/dev/null 2>&1; then ok=1; echo "✅ aiter 臂就绪（等待 $((i*10))s）"; break; fi
  if ! kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then echo "❌ 进程已退出"; break; fi
  [ $((i % 12)) -eq 0 ] && echo "  ...仍在等（$((i*10))s）：$(pgrep -c clang++ 2>/dev/null || echo 0) 个编译进程 / 构建 $(du -sh /home/qiba/ai/envs/vllm_0.28.0_rocm72/lib/python3.12/site-packages/aiter/jit/build/module_gemm_a8w8 2>/dev/null | cut -f1)"
  sleep 10
done

if [ "$ok" = "1" ]; then
  echo "--- 关键行（自证选了哪个内核）---"
  grep -aoE "Selected [A-Za-z0-9]*Int8ScaledMMLinearKernel|\[aiter\] import \[module_gemm_a8w8\][^\"]{0,40}|Loading weights took [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens.*|\[aiter\] \[module_gemm_a8w8\] prebuilt .so targets[^\"]{0,60}" "$SRV" | sort | uniq -c | sort -rn | head -8
  echo "--- 单流 TPS（256 token, greedy, MTP(1)，与 TRITON 臂同口径）---"
  python3 - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'ornith','prompt':'Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding.','max_tokens':256,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8115/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=1800)); dt=time.time()-t0
u=d['usage']; print('[int8-256k-aiter] SINGLE-STREAM %.2f tok/s (%d tok in %.1fs)'%(u['completion_tokens']/dt,u['completion_tokens'],dt))
PY
  curl -s localhost:8115/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{" | sed 's/^/  /'
  p=$(cat "$PIDF" 2>/dev/null || true)
  [ -n "${p:-}" ] && kill -TERM -"$p" 2>/dev/null && echo "已停 aiter 臂 $p"
  sleep 25
else
  echo "--- 未就绪，尾部日志 ---"
  grep -av "gfx90a-patch" "$SRV" | tail -6 | cut -c1-190
fi

echo "=== 转入 MTP 深度实验 ==="
bash "$REPO/mtp_depth.sh"
echo "=== aiter_finish + mtp_depth done $(date -u +%FT%TZ) ==="
