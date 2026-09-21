#!/usr/bin/env bash
# Hyperloom 8 小时会话：IR-2(install.sh) -> mi250x appliers -> 自检 -> IR-1(preflight) -> launch -> 健康检查
#
# 必须在容器内跑（Hyperloom 在宿主机不可导入；Magpie/vllm/aiter 都在容器 dist-packages；
# .env 是 root 0600）。宿主侧调用：
#   docker exec hyperloom-local bash /home/qiba/ROCm.AI/quark-int8/scripts_local/hl_session_flash8h.sh
#
# 为什么 install.sh 与 launch 在同一个脚本里：技能 IR-2 要求"跑 install.sh 并 source
# kernel-agent.env.sh 的那个 shell 就是 spawn optimize 的 shell"，拆成两次调用等于违约。
#
# 顺序固定（实测依据 hyperloom/reports/models/glm53-int4/hyperloom-mi250x-support-plan.md §9/§10）：
#   install.sh -> apply_mi250x_runner.py -> apply_mi250x_identity.py -> 起服
#   因为 install.sh 会动 Magpie，先打的 runner 注册会被盖掉。
set -uo pipefail

INSTALL_DIR=/home/qiba/ROCm.AI/hyperloom
SKILL_DIR=/home/qiba/ROCm.AI/amd-skills/skills/hyperloom-workload-optimizer
WORKLOAD_SRC=/home/qiba/ROCm.AI/quark-int8/scripts_local/hl_workload_flash8h.env
VPKG=/usr/local/lib/python3.12/dist-packages/vllm

bail() { echo "GATE_FAIL: $*"; exit 1; }

cd "$INSTALL_DIR" || bail "cd INSTALL_DIR"
export INSTALL_DIR SKILL_DIR
[ -f ./.env ] || bail ".env 不在 INSTALL_DIR"
set -a; . ./.env; set +a
: "${USER_DATA_PATH:?USER_DATA_PATH 缺失（先跑 hyperloom-setup）}"
export RUN_DIR="${USER_DATA_PATH}/optimizer_runs"
mkdir -p "$RUN_DIR"
export PYTHONPATH="${INSTALL_DIR}:${PYTHONPATH:-}"
ulimit -Sn 65536 || true

echo "== 0) 落库 workload.env（RUN_DIR 是 root 属主，只能在这里拷） =="
install -m 0644 "$WORKLOAD_SRC" "$RUN_DIR/workload.env" || bail "copy workload.env"
. "$RUN_DIR/workload.env"
echo "   MODEL=$MODEL_PATH"
echo "   TP=$TP EP=$EP CONC=$CONC ISL=$ISL OSL=$OSL PREC=$PRECISION HOURS=$MAX_HOURS GAIN=$TARGET_GAIN"
echo "   OPT_FLAGS=$OPT_FLAGS"
echo "   SERVER_ARGS=$SERVER_ARGS"

echo "== 1) 前置实物检查 =="
[ -f "$MODEL_PATH/config.json" ] || bail "模型在容器里不可见（挂载没生效？）"
[ -d "$SKILL_DIR/scripts" ] || bail "skill scripts 不在"
grep -q 'gfx90a-host patch' "$VPKG/models/glm5next/amd/sparse_indexer.py" || bail "indexer 放行件不在容器里"
grep -q 'if on_gfx90a():' "$VPKG/model_executor/layers/mhc.py" || bail "mHC gfx90a 回退件不在容器里"
python3 -c 'import vllm;print("   vllm="+vllm.__version__)' 2>/dev/null | tail -1

echo "== 2) IR-2：install.sh =="
INSTALL_SH="$INSTALL_DIR/hyperloom/inference_optimizer/assets/install.sh"
[ -f "$INSTALL_SH" ] || bail "install.sh 不在"
bash "$INSTALL_SH"
RC=$?
echo "INSTALL_RC=$RC"
[ $RC -eq 0 ] || bail "install.sh 非零退出"
KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${USER_DATA_PATH}/runtime/kernel-agent.env.sh}"
[ -f "$KERNEL_AGENT_ENV" ] || bail "kernel-agent.env.sh 缺失"
. "$KERNEL_AGENT_ENV"
export PYTHONPATH="${INSTALL_DIR}:${PYTHONPATH:-}"
echo "   PYTHON=${PYTHON:-$(command -v python3)}"

echo "== 3) 重打 mi250x 三处（install.sh 之后必须重来） =="
python3 "$INSTALL_DIR/patches-local/apply_mi250x_runner.py" || bail "apply_mi250x_runner"
python3 "$INSTALL_DIR/patches-local/apply_mi250x_identity.py" --install-dir "$INSTALL_DIR" || bail "apply_mi250x_identity"

echo "== 4) 静态自检（全 PASS 才继续） =="
python3 - <<'PY'
import pathlib, sys
hl = pathlib.Path("/home/qiba/ROCm.AI/hyperloom/hyperloom")
magpie = pathlib.Path("/usr/local/lib/python3.12/dist-packages/Magpie")
ok = True
def chk(name, cond):
    global ok
    print(("  PASS " if cond else "  FAIL ") + name)
    ok = ok and bool(cond)
gi = (hl / "common/gpu_identity.py").read_text()
chk("gpu_identity: mi250x -> (gfx90a, 104)", '"mi250x": ("gfx90a", 104)' in gi)
gt = (hl / "inference_optimizer/gpu_types.py").read_text()
chk("gpu_types: mi250x 不再折叠到 mi300x", 'if normalized in ("mi325x", "mi308x"):' in gt)
chk("gpu_types: _GFX_TO_RUNNER gfx90a", '"gfx90a": "mi250x"' in gt)
rc = (hl / "orchestrator/kernel/roofline_ceiling.py").read_text()
chk("roofline: HW_SPECS 含 mi250x", "mi250x" in rc)
chk("Magpie: vllm_mi250x.sh 在位", (magpie / "scripts/benchmark/vllm_mi250x.sh").is_file())
chk("Magpie: 已注册进 MAGPIE_BUILTIN_SCRIPTS", "vllm_mi250x.sh" in (magpie / "modes/benchmark/benchmarker.py").read_text())
chk("Magpie: image_selector gfx90a->mi250x", '"gfx90a": "mi250x"' in (magpie / "modes/benchmark/image_selector.py").read_text())
envs = pathlib.Path("/home/qiba/ROCm.AI/hyperloom/session/optimizer_runs/workload.env").read_text()
chk("workload.env: NUM_PROMPTS 钉到 16", "NUM_PROMPTS=16" in envs)
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || bail "静态自检未过"

# 探针必须用**非法值**触发 argparse 的 choices 报错，以此证明 mi250x 在合法集合里。
# 反面教材（2026-09-21 21:3x 本机实踩）：这里原来写的是"合法值 + 假模型路径"
#   optimize --gpu-type mi250x --model /tmp/nope-such-model
# 以为它会"报到模型路径错就退出"。实际链路是：argparse 全过 -> 会话目录真的建出来
# -> Claude 编排 agent 真的起来，跑了 3.5 分钟才发现并掐掉（session/nope-such-model/ 已删）。
# 报告里那条 V2 判据（"报错变成模型路径相关"）只在**没 source .env** 的裸环境下才安全；
# 一旦 .env 齐了，合法值探针 = 直接开一次真会话。
# 注意别被自己的 pipefail 咬：argparse 撞上非法 choices **必定 exit 2**（这正是我们要的
# 证据），所以 `python … | grep -q` 在 set -o pipefail 下会把 python 的 2 当成流水线状态，
# 明明匹配上了也判 FAIL（09-21 实踩过一次，误杀但没起会话）。先把输出收进变量，再按内容判定。
echo "== 5) CLI 接受性（非法值必须把 mi250x 列进 choices） =="
CLIOUT="$(python3 -m hyperloom.inference_optimizer.cli optimize --gpu-type __no_such_gpu__ --model /tmp/x --framework vllm 2>&1 | tail -2 || true)"
printf '%s\n' "$CLIOUT" | sed 's/^/   /'
case "$CLIOUT" in
  *mi250x*) echo "   mi250x 在 --gpu-type 的 choices 里：PASS" ;;
  *) bail "mi250x 不在 --gpu-type 的 choices 里（identity 补丁没生效？）" ;;
esac

# 5b) LLM 网关实地验证。09-21 21:3x 实踩：.env 的 ANTHROPIC_BASE_URL 写成
#   https://192.168.100.127:8107/anthropic  ——scheme 与路径**双错**（LiteLLM 只听
#   http://192.168.100.127:8107，且 /v1/messages 在根下）。后果不是报错而是**静默空转**：
#   会话目录建好、coordinator 活着、SEED turn 的 claude 子进程连不上网关就一直挂着，
#   8 小时预算会在零候选中烧完。所以这一门必须在 launch 之前，且要用**SDK 真正调用的
#   那个 bundled CLI**（不是裸 python 探针——裸探针只能证明 HTTP 可达，证不了 CLI 的
#   model/协议协商；不带 --model 时 CLI 还会偷偷用默认模型名去撞一个不存在的模型）。
echo "== 5b) LLM 网关实地验证 =="
CLAUDE_BIN="$(python3 -c 'import claude_agent_sdk,os;print(os.path.join(os.path.dirname(claude_agent_sdk.__file__),"_bundled","claude"))' 2>/dev/null | tail -1)"
[ -x "$CLAUDE_BIN" ] || bail "找不到 bundled claude CLI：$CLAUDE_BIN"
GWOUT="$(timeout 180 "$CLAUDE_BIN" -p 'reply with exactly: GATEWAY_OK' --model "${CLAUDE_MODEL:-}" 2>&1 | tail -3)"
echo "   gateway_reply=$GWOUT"
case "$GWOUT" in
  *GATEWAY_OK*) echo "   网关 PASS（base=$ANTHROPIC_BASE_URL model=$CLAUDE_MODEL）" ;;
  *) bail "LLM 网关不通：base=$ANTHROPIC_BASE_URL model=${CLAUDE_MODEL:-} —— 宁可不跑，不要空转 8 小时" ;;
esac

echo "== 6) IR-1：preflight =="
python3 "$SKILL_DIR/scripts/preflight.py" || bail "IR-1 preflight 未过"

echo "== 7) launch =="
bash "$INSTALL_DIR/../quark-int8/scripts_local/hl_launch.sh" || bail "hl_launch.sh"

echo "== 8) 30s 后健康检查 =="
sleep 30
bash "$SKILL_DIR/scripts/launch_health.sh"
echo "SESSION_LAUNCH_DONE"
