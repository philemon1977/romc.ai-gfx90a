#!/usr/bin/env bash
# =============================================================================
# 放开 hyperloom/session/ 里"该入库但读不到"的文件权限
#
# 背景：hyperloom 在容器里以 root 身份往本工作区写运行现场，留下 305 个
# root:root 0600 的文件（含全部 manifest.json / state.json /
# session_breakdown.json / reports/final.json）和 3 个 0700 的 runtime/ 目录。
# 当前用户读不到 → git 无法取哈希 → 这些 run 状态还没进仓库。
#
# 待办清单完全由 git 自己算（ls-files --others --exclude-standard），
# 因此本脚本不复制一份排除规则，不会与 .gitignore 漂移。
#
# 用法：
#   scripts/fix_session_perms.sh            # 只报告：读不到什么、多少、要执行什么
#   scripts/fix_session_perms.sh --apply    # 用 sudo 修权限（只动这些文件与 runtime/ 目录）
#
# 只加读权限（a+r / a+rX），不改属主、不动 worktree 与编译缓存那些不入库的东西。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -d hyperloom/session ]] || { echo "xx 没有 hyperloom/session/，无需处理" >&2; exit 0; }
command -v git >/dev/null || { echo "xx 需要 git" >&2; exit 1; }

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

# 1) git 认为"未跟踪且不被忽略"的 session 文件 = 还该入库的部分
git ls-files --others --exclude-standard -- hyperloom/session > "$tmp/pending" 2>"$tmp/warn" || true
# 2) 其中当前用户读不到的
python3 - "$tmp/pending" "$tmp/unreadable" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
rows=[]
with open(src, encoding='utf-8', errors='surrogateescape') as f:
    for line in f:
        p=line.rstrip('\n')
        if not p: continue
        try:
            with open(p,'rb') as fh: fh.read(1)
        except OSError:
            rows.append(p)
with open(dst,'w',encoding='utf-8',errors='surrogateescape') as o:
    o.write('\n'.join(rows) + ('\n' if rows else ''))
PY

read -r N MB < <(python3 -c "
import os
rows=[l.rstrip('\n') for l in open('$tmp/unreadable', encoding='utf-8', errors='surrogateescape') if l.strip()]
s=0
for p in rows:
    try: s+=os.path.getsize(p)
    except OSError: pass
print(len(rows), f'{s/1048576:.1f}')")

# 3) 0700 而进不去的目录（git 连文件名都列不出来）
python3 - hyperloom/session > "$tmp/undirs" <<'PY'
import os, sys
base = sys.argv[1]; out = []
for root, dirs, files in os.walk(base, onerror=lambda e: None):
    for d in list(dirs):
        p = os.path.join(root, d)
        try:
            os.listdir(p)          # 空目录返回 []；不会像 scandir().__next__() 那样抛 StopIteration
        except OSError:
            out.append(p)          # 只有真正进不去的才算
print('\n'.join(out))
PY

echo "hyperloom/session 待入库但读不到： $N 个文件, ${MB} MB"
echo "进不去的目录：                      $(grep -c . "$tmp/undirs" 2>/dev/null || echo 0) 个"
echo
if [[ -s "$tmp/unreadable" ]]; then
  echo "按名字统计（前 10）:"
  xargs -a "$tmp/unreadable" -r -n1 basename 2>/dev/null | sort | uniq -c | sort -rn | head -10
  echo
  echo "关键 run 状态是否在内:"
  for k in manifest.json state.json session_breakdown.json final.json optimization_journal.json specialist_done.json; do
    printf '  %-28s %s\n' "$k" "$(grep -c "/$k$" "$tmp/unreadable" || true) 个待放开"
  done
fi
echo
if [[ "$N" == "0" && ! -s "$tmp/undirs" ]]; then
  echo "没有需要修的。直接 git add hyperloom/session 即可补齐。"; exit 0
fi

if [[ "$APPLY" == "0" ]]; then
  echo "要执行（需要 root；本 harness 的 sudo 被 no new privileges 挡住，请在普通终端里跑）："
  echo
  echo "  git ls-files --others --exclude-standard -- hyperloom/session | sudo xargs -d '\n' -r chmod a+r"
  echo "  find hyperloom/session -type d ! -readable -exec sudo chmod a+rX {} +"
  echo
  echo "或者直接：  $0 --apply"
  exit 0
fi

echo "== 用 sudo 放开读权限 =="
if ! sudo -n true 2>/dev/null && ! sudo -v 2>/dev/null; then
  echo "xx sudo 不可用（当前会话被 no new privileges 限制）。请在普通终端里执行上面两条命令。" >&2
  exit 1
fi
xargs -a "$tmp/unreadable" -d '\n' -r sudo chmod a+r
[[ -s "$tmp/undirs" ]] && xargs -a "$tmp/undirs" -d '\n' -r sudo chmod a+rX

echo "== 复查 =="
left=$(git ls-files --others --exclude-standard -- hyperloom/session | while IFS= read -r f; do head -c1 "$f" >/dev/null 2>&1 || echo x; done | wc -l)
echo "仍读不到的待入库文件： $left"
if [[ "$left" == "0" ]]; then
  echo
  echo "权限已放开，补齐这一笔即可："
  echo "  git add hyperloom/session && git commit -m 'feat(session): 补齐 run 状态（manifest/state/session_breakdown）' && git push origin main"
fi
