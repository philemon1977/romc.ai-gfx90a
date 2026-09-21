#!/usr/bin/env python3
"""按适用范围（scope）从技能数据里筛选适用的知识条目。

为什么需要它
------------
`mi250x-recipe-ops` 的知识混在一处会互相毒化：一条在 vLLM 0.28.0 + bf16 + TP8 上测出来的
结论，被读到 llama.cpp Q4_K_M 臂上引用，就会得出错误动作。`data/scope.json` 定义了轴与词表，
`data/*.json` 的每条条目带 `applies_to` 谓词；**本脚本是这两个东西的执行体**——没有它，
分层只是文档，agent 照样错配。

与 `consumed_by` 的区别（最容易被混淆的一处）
---------------------------------------------
* `applies_to`   = 适用判定：九轴谓词，回答「换到我的目标上还成立吗」。**匹配用这个**。
* `consumed_by`  = 溯源/历史：真正消费过它的臂 id，回答「这东西在野外哪里被跑过」。
  它是**观测**，不是规则；实测 8121 被 0 个 knob 认领却实际适用其中两个（边单向），
  所以用 `consumed_by` 反查适用性会得到空集。`--orphans` 专门暴露这类缺口。

用法
----
  # 直接给目标九轴（能给的都给，给的越多判定越准；未给的轴 = unknown 不计入否决）
  python3 scripts/scope_match.py --engine vllm-0.28.0+rocm723 --quant int8-w8a8 \
      --topology tp1 --arch qwen3_5

  # 用某条现成服务臂的轴当作目标（想「把 8121 的经验搬到别处」时最常用）
  python3 scripts/scope_match.py --arm 8121-glm-5.3-ct-int4-w4a16-vllm-tp8

  # 列出目标轴与全部在册臂的差异（选型/迁移前的第一眼）
  python3 scripts/scope_match.py --arm 8114-... --diff-arms

  # 反向缺口：哪些臂没有被任何 knob 声明适用（真实发生过，见 --orphans 说明）
  python3 scripts/scope_match.py --orphans

  # 打印九轴词表
  python3 scripts/scope_match.py --axes

退出码：0 = 有适用条目；1 = 没有任何条目适用（或 --orphans 发现缺口）；2 = 输入不可用。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SKILL = Path("/home/qiba/ROCm.AI/local-skills/mi250x-recipe-ops")
DATA = SKILL / "data"
FILES = [
    ("arms.json", "arms", "recipe_id"),
    ("environments.json", "environments", "id"),
    ("patches.json", "patches", "id"),
    ("knobs.json", "knobs", "id"),
    ("ops.json", "ops", "id"),
]
AXES = ["host", "driver", "rocm", "torch", "engine", "arch", "model", "quant", "topology"]

# serving 配方 frontmatter 的 `env:` 写的是**目录**（`envs/xxx`，llama.cpp 还带子前缀），
# 而 environment 配方的 id 没有 `envs/` 前缀 ⇒ 直接比对是 0/15 可解析。
# 归一化（去前缀 + 取首段 + 特例）后 15/15。特例只有一条：8121 的环境实体是镜像 tag，
# `env:` 写的是挂载进镜像的补丁树路径。
ENV_SPECIAL = {
    "patches/gfx90a/ct_w4a16_dsv41_n0918": "vllm-openai-rocm-nightly-0918",
}
AI_RECIPES = Path("/home/qiba/ai/docs/recipes")


def resolve_env(v: str, env_ids):
    v = (v or "").strip().strip('"\'')
    if v in ENV_SPECIAL:
        return ENV_SPECIAL[v]
    v = re.sub(r"^envs/", "", v)
    head = v.split("/")[0]
    return head if head in env_ids else None


def _fm_env(md: Path):
    """从配方 markdown 的 frontmatter 里取 `env:` 原值（只需一行式，别引 YAML）。"""
    if not md.is_file():
        return None
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", md.read_text(encoding="utf-8"), re.S)
    if not m:
        return None
    k = re.search(r"^env:\s*(.+)$", m.group(1), re.M)
    return k.group(1).strip() if k else None


def env_check(items) -> int:
    """跨表一致性：**臂的版本轴必须与它所引用环境的版本轴一致**。

    这是九轴里 `rocm`/`torch` 两轴唯一可机器验证的地方——`config/env-lock/*.txt`
    一个 ROCm 版本字符都不记（实测 8 个文件 0 命中），版本三元组只能由
    env-lock + config/ais.env + tools/build_*.sh 三方拼，拼完就可能拼错。
    本检查把它钉住：join 不通、或轴值与 env 配方冲突 ⇒ 失败。
    """
    arms = [x for x in items if x["file"] == "arms.json"]
    envs = {x["id"]: x for x in items if x["file"] == "environments.json"}
    bad = 0
    print("臂 → 环境 的版本轴一致性（rocm / torch / engine）")
    for it in sorted(arms, key=lambda x: x["id"]):
        md = AI_RECIPES / "serving" / f"{it['id']}.md"
        raw = _fm_env(md)
        if raw is None:
            print(f"  ⚠ {it['id']}: 找不到配方或没有 `env:` 字段（{md}）")
            bad += 1
            continue
        eid = resolve_env(raw, envs)
        if eid is None:
            print(f"  ❌ {it['id']}: `env: {raw}` 解析不到 environment 配方")
            bad += 1
            continue
        ea, ia = envs[eid]["applies_to"], it["applies_to"]
        # 用与匹配器同一套语义（列表成员 + glob），**不能**用字符串相等比较：
        # 一棵 llama.cpp 树里含多个 commit 子前缀 ⇒ env 侧是 list，臂侧是 str，
        # 直接 str() 相比会造出 4 处假冲突（2026-09-21 实测踩过）。
        diffs = []
        for ax in ("rocm", "torch", "engine"):
            if ax not in ia or ax not in ea:
                continue
            a_vals = [x for x in as_list(ia[ax]) if x != "*"]
            if not a_vals:
                continue
            e_vals = as_list(ea[ax])
            if not any(_hit(ev, av) for av in a_vals for ev in e_vals):
                diffs.append(f"{ax}: 臂={ia[ax]} 不在 env={ea[ax]} 内")
        if diffs:
            bad += 1
            print(f"  ❌ {it['id']} → {eid}")
            for d in diffs:
                print(f"       {d}")
        else:
            print(f"  ✓ {it['id']:52s} → {eid:34s} "
                  f"rocm={ea.get('rocm')} torch={str(ea.get('torch'))[:20]}")
    if bad:
        print(f"\n❌ {bad} 处不一致/解析失败")
        return 1
    print(f"\n✓ {len(arms)}/{len(arms)} 条臂的 env 可解析且版本轴与所引用环境一致")
    return 0


def load():
    spec = json.loads((DATA / "scope.json").read_text(encoding="utf-8"))
    presets = {k: v for k, v in spec.get("scope_presets", {}).items()
               if not k.startswith("_")}
    items = []
    for fname, key, idf in FILES:
        j = json.loads((DATA / fname).read_text(encoding="utf-8"))
        for e in j[key]:
            items.append({
                "file": fname,
                "kind": key[:-1] if key.endswith("s") else key,
                "id": e.get(idf) or e.get("id"),
                "title": (e.get("title_or_why") or e.get("recipe_id") or "")[:110],
                "applies_to": expand(e.get("applies_to") or {}, presets),
                "consumed_by": e.get("consumed_by") or [],
                "effect_by_scope": e.get("effect_by_scope") or [],
                "status": e.get("status", ""),
                # 跨表一致性检查要用：臂引用的环境、以及配方 frontmatter 里的 env 原值
                "env_path": e.get("env_path") or e.get("path_or_target") or "",
                "recipe_env": e.get("_recipe_env", ""),
            })
    return spec, items


def expand(applies, presets):
    """展开 preset；显式给的轴覆盖 preset 的同名轴。"""
    if "preset" not in applies:
        return applies
    name = applies["preset"]
    if name not in presets:
        raise SystemExit(f"未知 preset: {name}（scope.json > scope_presets）")
    out = dict(presets[name])
    out.update({k: v for k, v in applies.items() if k != "preset"})
    return out


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _hit(pattern: str, value: str) -> bool:
    """模式匹配：`*` = 任意；`vllm-*` = 前缀 glob；其余全等。

    glob 是必需的，不是便利：跨臂开关（QuickReduce / MTP / 工具解析器）的规则
    对**任何** vLLM 版本都成立，差别在**效果**（那是 effect_by_scope 的活）。
    把这类开关钉死在某一个引擎版本上，会让"换版本后还能不能用"这个问题
    得到错误的否定答案——第一次跑就是这个毛病。
    """
    if pattern == "*":
        return True
    if "*" in pattern:
        import fnmatch
        return fnmatch.fnmatchcase(value, pattern)
    return pattern == value


def match(item, target):
    """返回 (verdict, reasons)。verdict ∈ {applies, no, unknown}。"""
    reasons, unknown = [], 0
    for axis, want in (item["applies_to"] or {}).items():
        if axis in ("tier",):          # tier 是分层标签，不参与目标匹配
            continue
        wants = as_list(want)
        if any(w == "*" for w in wants):
            continue
        if axis not in target or not target[axis]:
            unknown += 1
            continue
        got = as_list(target[axis])
        if not any(_hit(w, g) for w in wants for g in got):
            reasons.append(f"{axis}: 需要 {wants}，目标 {got}")
    if reasons:
        return "no", reasons
    if unknown:
        return "unknown", [f"{unknown} 个轴未给出（不否决，但判定不完整）"]
    return "applies", []


def fmt(item, verdict, reasons):
    mark = {"applies": "✅", "unknown": "❓", "no": "❌"}[verdict]
    a = item["applies_to"]
    sc = " ".join(f"{k}={v}" for k, v in sorted(a.items())
                  if k != "tier" and v != "*")
    line = f"  {mark} [{a.get('tier','?')}] {item['id']}\n       scope: {sc or '(全轴通配)'}"
    if reasons:
        line += "\n       " + "；".join(reasons)
    if item["effect_by_scope"]:
        for eb in item["effect_by_scope"]:
            w = ", ".join(f"{k}={v}" for k, v in (eb.get("when") or {}).items()) or "任意"
            line += f"\n       ↳ 效果[{w}] = {eb['effect']}：{eb.get('value','')[:150]}"
    return line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for ax in AXES:
        ap.add_argument(f"--{ax}", action="append", default=[], help=f"目标 {ax} 轴（可重复）")
    ap.add_argument("--arm", help="用该臂的 applies_to 当目标")
    ap.add_argument("--diff-arms", action="store_true", help="列出目标与其它臂的轴差异")
    ap.add_argument("--orphans", action="store_true", help="找被 0 个 knob 声明适用的臂")
    ap.add_argument("--env-check", action="store_true",
                    help="跨表一致性：臂的版本轴必须与它所引用环境的一致")
    ap.add_argument("--axes", action="store_true", help="打印九轴词表")
    ap.add_argument("--kind", action="append", default=[], help="只看某类（arms/knobs/...）")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="不适用项也逐条打印否决轴（默认只列 id）")
    args = ap.parse_args()

    spec, items = load()

    if args.axes:
        print(f"# {spec['title']}  (schema v{spec['schema_version']})")
        print("\n## 分层")
        for t, d in spec["tiers"].items():
            print(f"  {t}  {d['name']}\n      {d['rule']}\n      失效于：{d['invalidated_by']}")
        print("\n## 轴与词表")
        for ax, d in spec["axes"].items():
            print(f"\n  {ax} — {d['question']}")
            for vid, vd in d["vocab"].items():
                print(f"      {vid:34s} {vd['label']}")
        return 0

    if args.env_check:
        return env_check(items)

    if args.orphans:
        arms = {i["id"]: i for i in items if i["file"] == "arms.json"}
        knobs = [i for i in items if i["file"] == "knobs.json"]
        print("臂 ← knob 覆盖：`观测` = consumed_by 里有它（历史事实）；"
              "`谓词` = applies_to 判定适用（规则）\n")
        hard = soft = 0
        for aid in sorted(arms):
            obs = {k["id"] for k in knobs if aid in (k["consumed_by"] or [])}
            pre = {k["id"] for k in knobs if match(k, arms[aid]["applies_to"])[0] == "applies"}
            missed = pre - obs          # 适用但没人记录用过
            contra = obs - pre          # 记录了用过，谓词却说不适用 ← 危险
            flags = []
            if contra:
                flags.append(f"🛑 矛盾 {len(contra)}")
                hard += 1
            if missed:
                flags.append(f"▫️ 未记录 {len(missed)}")
                soft += 1
            print(f"  {aid}")
            print(f"      观测 {len(obs):2d} / 谓词 {len(pre):2d}   {'  '.join(flags) or '✓ 一致'}")
            if contra:
                print(f"      🛑 **记录了用过但判不适用**（要么 consumed_by 错，要么 applies_to 太窄）："
                      f"{sorted(contra)}")
            if missed and args.verbose:
                print(f"      ▫️ 适用但 consumed_by 未记（可能没试过，也可能只是没回填）：{sorted(missed)}")
        print(f"\n合计：{hard} 条臂有**矛盾**（必须查），{soft} 条臂有**未记录**（回填或标注未试）。")
        if hard:
            print("矛盾比缺口严重：它意味着历史记录与适用规则打架，照任一边行动都可能错。")
            return 1
        return 0

    target: dict[str, list[str]] = {}
    label = ""
    if args.arm:
        arms = {i["id"]: i for i in items if i["file"] == "arms.json"}
        if args.arm not in arms:
            print(f"没有这条臂：{args.arm}\n可选：\n  " + "\n  ".join(sorted(arms)),
                  file=sys.stderr)
            return 2
        target = {k: as_list(v) for k, v in arms[args.arm]["applies_to"].items()}
        label = args.arm
    for ax in AXES:
        for v in getattr(args, ax):
            target.setdefault(ax, [])
            if v not in target[ax]:
                target[ax].append(v)

    if not target:
        print("没有给任何目标轴。用 --arm <id> 或 --engine/--quant/… 指定；--help 看用法。",
              file=sys.stderr)
        return 2

    print(f"目标 scope：{label or '(命令行给出)'}")
    for ax in AXES:
        if target.get(ax):
            print(f"  {ax:9s} = {', '.join(as_list(target[ax]))}")
    missing = [a for a in AXES if not target.get(a)]
    if missing:
        print(f"  未指定（不否决，判定不完整）：{', '.join(missing)}")
    print()

    if args.diff_arms:
        arms = sorted((i for i in items if i["file"] == "arms.json"), key=lambda x: x["id"])
        print("与其它臂的轴差异（只列不同轴）")
        for a in arms:
            if a["id"] == label:
                continue
            diffs = []
            for ax in AXES:
                t = set(as_list(target.get(ax)))
                o = set(as_list(a["applies_to"].get(ax)))
                if t and o and not (t & o):
                    diffs.append(f"{ax}: {sorted(t)} → {sorted(o)}")
            if diffs:
                print(f"\n  {a['id']}")
                for d in diffs:
                    print(f"      {d}")
        print()

    buckets = {"applies": [], "unknown": [], "no": []}
    # --kind 单复数都收：kind 字段是单数（knob/patch/…），文件名是复数（knobs.json）
    want_kinds = {k.rstrip("s") if k.endswith("s") and k != "ops" else k
                  for k in args.kind}
    want_kinds |= {k for k in args.kind}
    for it in items:
        if args.kind and not ({it["kind"], it["file"], it["file"].removesuffix(".json")}
                              & want_kinds):
            continue
        if label and it["id"] == label:
            continue
        v, r = match(it, target)
        buckets[v].append((it, r))

    for v, head in (("applies", "✅ 适用"), ("unknown", "❓ 判定不完整（有轴未指定）"),
                    ("no", "❌ 不适用")):
        print(f"\n{'='*72}\n{head}  ({len(buckets[v])})\n{'='*72}")
        if v == "no" and not args.verbose:
            # 默认只列 id + 第一个否决轴：51 条逐条打全会把有用的淹掉。
            for it, r in buckets[v]:
                first = r[0].split("：")[0] if r else ""
                print(f"  ❌ {it['id']}   (否决轴: {first})")
            if buckets[v]:
                print("  … 加 -v 看每条的完整否决理由")
            continue
        for it, r in buckets[v]:
            print(fmt(it, v, r))

    print(f"\n{'='*72}")
    print(f"结论：{len(buckets['applies'])} 条确定适用 / "
          f"{len(buckets['unknown'])} 条判定不完整 / {len(buckets['no'])} 条不适用。")
    if buckets["unknown"]:
        print("补上缺失的轴可以消除「判定不完整」——它们既不是适用也不是不适用。")
    return 0 if buckets["applies"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
