#!/usr/bin/env bash
# =============================================================================
# 放开 hyperloom/session/ 里"该入库但读不到"的文件权限
#
# 背景：hyperloom 在容器里以 root 身份往本工作区写运行现场，留下 root:root 0600
# 的文件（曾含全部 manifest.json / state.json / session_breakdown.json）和 0700 的
# runtime/ 目录。当前用户读不到 → git 无法取哈希 → 这些 run 状态进不了仓库。
#
# 待办清单完全由 git 自己算（ls-files --others --exclude-standard），所以本脚本
# 不复制一份排除规则，不会与 .gitignore 漂移；已被 .gitignore 排除的东西（例如
# runtime/ 里那个带明文密钥的 env 脚本）不会被列进来，也不会被 chmod。
#
# 用法：
#   scripts/fix_session_perms.sh            # 只报告
#   scripts/fix_session_perms.sh --apply    # 用 sudo 修（迭代到不动点）
#
# 只加读位（a+r / a+rX），不改属主，不动不入库的东西。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -d hyperloom/session ]] || { echo "没有 hyperloom/session/，无需处理"; exit 0; }
command -v git >/dev/null || { echo "需要 git" >&2; exit 1; }

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

list_pending()    { git ls-files --others --exclude-standard -- hyperloom/session > "$tmp/pending" 2>/dev/null || true; }
list_unreadable() {
  python3 - "$tmp/pending" "$tmp/unreadable" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
rows = []
with open(src, encoding='utf-8', errors='surrogateescape') as f:
    for line in f:
        p = line.rstrip('\n')
        if not p:
            continue
        try:
            with open(p, 'rb') as fh:
                fh.read(1)
        except OSError:
            rows.append(p)
with open(dst, 'w', encoding='utf-8', errors='surrogateescape') as o:
    o.write('\n'.join(rows) + ('\n' if rows else ''))
PY
}
# 0700 的目录里藏着 git 根本列不出来的文件：先开目录，再谈文件。
# 用 os.listdir 而不是 scandir().__next__()——后者在空目录上抛 StopIteration。
list_undirs() {
  python3 - hyperloom/session > "$tmp/undirs" <<'PY'
import os, sys
out = []
for root, dirs, _ in os.walk(sys.argv[1], onerror=lambda e: None):
    for d in list(dirs):
        p = os.path.join(root, d)
        try:
            os.listdir(p)
        except OSError:
            out.append(p)
print('\n'.join(out))
PY
}
refresh() { list_pending; list_unreadable; list_undirs; }

count_and_size() {
  python3 -c "
import os,sys
rows=[l.rstrip('\n') for l in open(sys.argv[1], encoding='utf-8', errors='surrogateescape') if l.strip()]
s=0
for p in rows:
    try: s+=os.path.getsize(p)
    except OSError: pass
print(len(rows), f'{s/1048576:.1f}')" "$1"
}

refresh
N=0; MB=0; read -r N MB < <(count_and_size "$tmp/unreadable")
ND=$(grep -c . "$tmp/undirs" || true)   # grep -c 无匹配时自己就打印 0，不能再 || echo
TOTAL=$(wc -l < "$tmp/pending")

echo "hyperloom/session 待入库：       $TOTAL 个"
echo "其中读不到：                     $N 个, ${MB} MB"
echo "进不去的目录（里面还藏着文件）： $ND 个"
echo

if [[ "$N" != "0" ]]; then
  echo "读不到的按名字统计（前 10）:"
  xargs -a "$tmp/unreadable" -r -n1 basename 2>/dev/null | sort | uniq -c | sort -rn | head -10
  echo
  echo "关键 run 状态:"
  for k in manifest.json state.json session_breakdown.json final.json optimization_journal.json specialist_done.json; do
    printf '  %-26s %s 个\n' "$k" "$(grep -c "/$k$" "$tmp/unreadable" || true)"
  done
  echo
fi

if [[ "$N" == "0" && "$ND" == "0" ]]; then
  echo "没有需要修的。补齐仓库："
  echo "  git ls-files --others --exclude-standard -- hyperloom/session | xargs -d '\n' -r git add --"
  echo "（别用 \`git add hyperloom/session\` 整目录写法，理由见 DEPENDENCIES.md 第 9 节）"
  exit 0
fi

if [[ "$APPLY" == "0" ]]; then
  echo "要执行（需要 root；本 harness 的 sudo 被 no new privileges 挡住，请在普通终端里跑）："
  echo
  echo "  ./scripts/fix_session_perms.sh --apply"
  echo
  echo "等价的裸命令（注意：一趟不够，打开 0700 目录会暴露出新一批文件，需迭代到不动点）："
  echo "  find hyperloom/session -type d ! -readable -exec sudo chmod a+rX {} +"
  echo "  git ls-files --others --exclude-standard -- hyperloom/session | sudo xargs -d '\n' -r chmod a+r"
  exit 0
fi

echo "== 用 sudo 放开读位，迭代到不动点 =="
if ! sudo -n true 2>/dev/null && ! sudo -v 2>/dev/null; then
  echo "xx sudo 不可用（当前会话被 no new privileges 限制）。请在普通终端里重跑本脚本。" >&2
  exit 1
fi

prev=-1
for round in 1 2 3 4; do
  [[ -s "$tmp/undirs" ]] && xargs -a "$tmp/undirs" -d '\n' -r sudo chmod a+rX
  [[ -s "$tmp/unreadable" ]] && xargs -a "$tmp/unreadable" -d '\n' -r sudo chmod a+r
  refresh
  read -r N MB < <(count_and_size "$tmp/unreadable")
  ND=$(grep -c . "$tmp/undirs" || true)   # grep -c 无匹配时自己就打印 0，不能再 || echo
  echo "   第 $round 轮后：读不到 $N 个 / 进不去 $ND 个目录"
  cur=$((N + ND))
  [[ "$cur" == "$prev" ]] && { echo "   已收敛。"; break; }
  prev=$cur
  [[ "$cur" == "0" ]] && break
done

echo
if [[ "$N" == "0" && "$ND" == "0" ]]; then
  echo "权限已放开。补齐仓库（显式 pathspec，不用整目录 add）："
  echo "  git ls-files --others --exclude-standard -- hyperloom/session | xargs -d '\n' -r git add --"
  echo "  git commit -m 'feat(session): 补齐 run 状态' && git push origin main"
else
  echo "仍有 $N 个文件 / $ND 个目录读不到。若它们落在 hyperloom/session/**/runtime/，"
  echo "那是**有意排除**的（内含明文 ANTHROPIC_API_KEY，见 DEPENDENCIES.md 第 9 节），不要修权限。"
  echo "其余的属主/权限情况：$(head -5 "$tmp/unreadable" | xargs -r ls -l 2>/dev/null | head -3)"
fi
