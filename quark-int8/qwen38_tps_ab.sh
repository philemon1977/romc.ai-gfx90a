#!/usr/bin/env bash
# 8107 单流提速 A/B（目标 ≥80 t/s）：① 出厂默认（已测 92.07/71.32）② 关前缀缓存 ③ SPEC=2/4 重扫
# 纪律：固定 prompt + 丢弃首个请求 + 中位数 n=5；同时报接受率与 step ms；功率档 560 W 由 launcher 护栏④校验。
set -uo pipefail
LAUNCH=/mnt/stripe-3mix-3t2/models/Qwen/Qwen3.8-Flash-Next-BF16/../launcher/qwen3.8-flash-next_176b_bf16_vllm_rocm724_256k_8107_Qwen_mi250dx8.sh
LAUNCH=/mnt/stripe-3mix-3t2/models/Qwen/launcher/qwen3.8-flash-next_176b_bf16_vllm_rocm724_256k_8107_Qwen_mi250dx8.sh
PIDF=/home/qiba/ai/logs/qwen3.8-flash-next-8107.pid
REPO=/home/qiba/ROCm.AI/quark-int8
exec > >(tee -a "$REPO/logs/qwen38_tps_ab.log") 2>&1
echo "=== 8107 单流提速 A/B  $(date -u +%FT%TZ) ==="

stop_mine () {
  [ -f "$PIDF" ] || { echo "  （无 PID 文件——不是本脚本起的，不动）"; return 0; }
  local p; p=$(cat "$PIDF" 2>/dev/null || true); rm -f "$PIDF"
  [ -n "${p:-}" ] || return 0
  kill -0 "$p" 2>/dev/null || { echo "  $p 已不在（陈旧归档）"; return 0; }
  if ! tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--port 8107"; then
    echo "  ⛔ pid $p 不是 8107 服务，拒绝杀"; return 1
  fi
  echo "  停本会话 8107 服务 pid=$p（TERM → 25s → KILL；本底座实测 TERM 会挂住）"
  kill -TERM -"$p" 2>/dev/null
  local i; for i in $(seq 1 5); do kill -0 "$p" 2>/dev/null || break; sleep 5; done
  if kill -0 "$p" 2>/dev/null; then
    echo "  ⚠️ TERM 25s 未生效 ⇒ 升级 KILL（进程组）"
    kill -KILL -"$p" 2>/dev/null; sleep 5
  fi
  if kill -0 "$p" 2>/dev/null; then echo "  ⛔ KILL 后仍存活，拒绝对 GPU 做任何事"; return 1; fi
  # 兜底：worker 若残留（仅在本会话独占窗口内、且已确认无别人的 api_server 时）
  if [ "$(pgrep -cf 'vllm.entrypoints.openai.api_server' || true)" -le 1 ]; then
    local w; for w in $(pgrep -f "VLLM::Worker_TP" 2>/dev/null || true); do
      echo "    清残留 worker pid=$w"; kill -KILL "$w" 2>/dev/null
    done
  fi
  return 0
}
wait_released () {
  local i free wk
  for i in $(seq 1 90); do
    nc -z 127.0.0.1 8107 2>/dev/null && { sleep 5; continue; }
    wk=$(rocm-smi --showpids 2>/dev/null | grep -c "VLLM::Worker" || true)
    free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
    if [ "${free:-0}" -ge 58 ] && [ "${wk:-1}" = "0" ]; then
      echo "  已释放（min free ${free} GiB，残留 worker ${wk}，$((i*5))s）"; return 0
    fi
    sleep 5
  done
  echo "  ⚠️ 释放等待超时（min free ${free:-?} GiB，残留 worker ${wk:-?}）"; return 1
}
measure () {  # tag
  local tag=$1
  python3 - "$tag" <<'PY'
import json,re,statistics,sys,time,urllib.request
tag=sys.argv[1]; P=8107; M="qwen3.8-flash-next"
CNT="Count slowly from one to ninety, writing each number in words on its own line."
EXP=("Explain in detail why int4 quantization reduces memory bandwidth pressure during decoding, "
     "covering weights and KV cache.")
def scrape():
    t=urllib.request.urlopen(f"http://127.0.0.1:{P}/metrics",timeout=60).read().decode(); o={}
    for k in ("spec_decode_num_draft_tokens","spec_decode_num_accepted_tokens"):
        m=re.search(rf"^vllm:{k}_total\{{[^}}]*\}}\s+([0-9.eE+]+)$",t,re.M); o[k]=float(m.group(1)) if m else 0.0
    return o
def one(p):
    b=json.dumps({"model":M,"prompt":p,"max_tokens":256,"temperature":0,"ignore_eos":True}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{P}/v1/completions",data=b,headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.load(urllib.request.urlopen(r,timeout=3600)); return time.time()-t0, d["usage"]["completion_tokens"]
for name,p in (("count",CNT),("explain",EXP)):
    one(p)
    c0=scrape(); ts=[]
    for _ in range(5):
        dt,n=one(p); ts.append(n/dt)
    c1=scrape()
    dd=c1["spec_decode_num_draft_tokens"]-c0["spec_decode_num_draft_tokens"]
    da=c1["spec_decode_num_accepted_tokens"]-c0["spec_decode_num_accepted_tokens"]
    med=statistics.median(ts); acc=da/dd if dd else 0.0
    spec=3 if dd else 0
    tps_step=1+spec*acc
    steps=1280/tps_step if tps_step else float("nan")
    stepms=1000*statistics.median(ts)*0+1000*(statistics.median([256/x for x in ts]))/steps
    print(f"[{tag}] {name:7s} MEDIAN {med:6.2f} t/s  min {min(ts):.2f} max {max(ts):.2f} "
          f"spread {100*(max(ts)-min(ts))/med:.1f}%  | 接受 {100*acc:.1f}% | tok/step {tps_step:.2f} "
          f"| step {stepms:.1f} ms", flush=True)
PY
}

run_arm () {  # tag extra
  local tag=$1 extra=$2
  echo; echo "######## ARM $tag  EXTRA='$extra'  $(date -u +%H:%M:%S) ########"
  stop_mine; wait_released || return 1
  # ⚠️ 判据必须排除"僵尸"：zombie 的 /proc/<pid>/cmdline 仍在，pgrep -f 照样命中，
  #    于是刚被 KILL 掉、还没被回收的服务会被误判成"别人在跑"（本会话实撞：4 个臂被瞬间跳过）。
  _others=""; _zomb=""
  for _p in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true); do
    [ "$_p" = "$$" ] && continue
    _st=$(awk '{print $3}' /proc/$_p/stat 2>/dev/null || echo "")
    if [ -z "$_st" ]; then continue; fi
    if [ "$_st" = "Z" ]; then _zomb="$_zomb $_p"; else _others="$_others $_p"; fi
  done
  [ -n "$_zomb" ] && echo "  （忽略僵尸进程：$_zomb —— 已死未回收，不占 GPU）"
  if [ -n "$_others" ]; then
    echo "  ⛔ 还有活的 vLLM 进程：$_others（可能是别的会话）⇒ 不抢，退出"; return 1
  fi
  if [ -n "$extra" ]; then export VLLM_EXTRA_ARGS="$extra"; else unset VLLM_EXTRA_ARGS; fi
  PORT=8107 bash "$LAUNCH" || { echo "[$tag] ❌ LAUNCH FAILED"; return 1; }
  local i
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf http://127.0.0.1:8107/health >/dev/null 2>&1 || { echo "[$tag] ❌ NOT READY"; return 1; }
  echo "  ✅ ready"
  L=$(cat /home/qiba/ai/logs/qwen3.8-flash-next-8107.logpath 2>/dev/null)   # 2026-09-18：日志改落 logs/flash-next/，认侧车而不是 glob（旧 glob 会静默读到旧日志）
[ -n "$L" ] && [ -r "$L" ] || L=$(ls -t /home/qiba/ai/logs/qwen3.8-flash-next_*8107*.log /home/qiba/ai/logs/flash-next/server-8107-*.log 2>/dev/null | head -1)
  grep -aoE "enable_prefix_caching=[A-Za-z]+|Mamba cache mode is set to '[a-z]+'|GPU KV cache size: [0-9,]+ tokens" "$L" | sort -u | sed 's/^/    /'
  measure "$tag"
}

# ① 关前缀缓存（P1）
run_arm nopfx "--no-enable-prefix-caching"
# ② SPEC=2 / ④ SPEC=4 重扫（P2；定稿口径）
run_arm spec2 "--no-enable-prefix-caching --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}"
run_arm spec4 "--no-enable-prefix-caching --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":4}"
# ③ 收尾：恢复出厂默认（前缀缓存开、SPEC=3）
echo; echo "######## 收尾：恢复出厂默认 ########"
run_arm default ""
echo "=== done $(date -u +%FT%TZ) ==="
