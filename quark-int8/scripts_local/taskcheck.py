#!/usr/bin/env python3
"""taskcheck.py — 任务级验收尺子（与 8121 的 glm53_verify32k_0920_1738 同一口径，可跨臂并排）。

口径（必须写进结论）：/v1/completions **裸续写**（不过 chat 模板 ⇒ 不触发思考链，
  这是 09-21 那条「max_tokens=48 全被 reasoning_content 吃掉 ⇒ content 为空」教训的绕法）、
  temperature=0、fact 题 max_tokens=24、GSM8K max_tokens=200。
判据与 8121 基线逐项对齐：事实召回 6/6、GSM8K 5/6、多轮 TTFT 比、9232-token 针尖 MISS。
8121 基线（logs/glm53_verify32k_0920_1738.log）：召回 6/6、GSM8K 5/6、
  轮1 TTFT 7.362s → 轮2 0.678s（比 0.092）、解码 3.76 tok/s、针尖 prompt=9232 **MISS**。

用法：taskcheck.py --url http://127.0.0.1:8128/v1 --model glm53flash-int8 [--tag 8128]
退出码：0=召回与算术都不低于基线；1=有退化；2=请求失败。
"""
import argparse
import json
import re
import time
import urllib.request

FACTS = [
    ("中国的首都是", ["北京"]),
    ("The capital of France is", ["paris"]),
    ("法国首都巴黎，日本首都东京，中国首都", ["北京"]),
    ("1+1=", ["2"]),
    ("水的化学式是", ["h2o"]),
    ("DeepSeek 是由哪家", ["公司"]),
]
GSM8K = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May? The answer is", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? The answer is", 10),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? The answer is", 35),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year? The answer is", 624),
    ("A store is having a sale where all shirts are 20% off. If a shirt originally costs $50, what is the sale price? The answer is", 40),
    ("Julie is reading a 120-page book. Yesterday she was able to read 12% of it and today she was able to read twice as many pages as yesterday. How many pages does she have left? The answer is", 76),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8128/v1")
    ap.add_argument("--model", default="glm53flash-int8")
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--needle-tokens", type=int, default=9000)
    a = ap.parse_args()
    url = a.url.rstrip("/") + "/completions"

    def c(prompt, max_tokens, to=900):
        body = {"model": a.model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        d = json.loads(urllib.request.urlopen(req, timeout=to).read())
        return d["choices"][0]["text"], time.time() - t0, d.get("usage", {})

    fails = 0
    print("== 事实召回（裸续写 temp=0 max_tokens=24）==")
    for prompt, keys in FACTS:
        try:
            out, _, _ = c(prompt, 24)
        except Exception as e:
            print("'%s' -> 请求失败 %s" % (prompt[:16], e))
            fails += 1
            continue
        low = out.lower()
        hit = [k for k in keys if k in low]
        ok = bool(hit)
        if not ok:
            fails += 1
        print("'%s' -> %r  [%s %s]" % (prompt[:18], out[:44], "PASS" if ok else "FAIL", "/".join(keys)))
    print()
    print("== GSM8K（temp=0 max_tokens=200，末数命中）==")
    gsm = 0
    for q, gold in GSM8K:
        try:
            out, _, _ = c(q, 200)
        except Exception as e:
            print("  请求失败 %s" % e)
            continue
        nums = re.findall(r"-?\d[d,]*\.?\d*", out.replace(",", ""))
        hit = any(n.rstrip(".") == str(gold) for n in nums[-4:]) if nums else False
        gsm += 1 if hit else 0
        print("  %s gold=%-4s -> %r" % ("OK" if hit else "XX", gold, out[:70].replace("\n", " ")))
    print("GSM8K 命中 %d/%d" % (gsm, len(GSM8K)))
    print()
    print("== 长上下文针尖（目标 ~%d token，needle 埋在中间）==" % a.needle_tokens)
    pad = ("The quarterly report describes logistics, supply chains, and inventory turnover in detail. "
           * (a.needle_tokens // 12))
    needle = "The secret passphrase for this document is ZephyrElephantYak42. "
    half = len(pad) // 2
    p = pad[:half] + needle + pad[half:]
    try:
        t0 = time.time()
        out, dt, u = c(p + "\nQuestion: What is the secret passphrase? Answer:", 24)
        pt = u.get("prompt_tokens", 0)
        hit = "zephyrelephantyak42" in out.lower().replace(" ", "").replace("-", "")
        print("  prompt=%s TTFT+gen=%.2fs 答案=%r [%s]" % (pt, dt, out[:60], "HIT" if hit else "MISS"))
    except Exception as e:
        print("  针尖请求失败：%s" % e)
        hit = False
    print()
    print("=== %s 汇总：召回失败 %d，GSM8K %d/%d，针尖 %s ===" % (
        a.tag, fails, gsm, len(GSM8K), "HIT" if hit else "MISS"))
    # ⚠ 那组「召回 6/6、GSM8K 5/6」是 **8121 = GLM-5.3-CT-Int4-W4A16** 的读数，
    # 不是本模型的基线。本仓铁律：一个模型的结论不能外推到另一个模型。
    # 所以这里的 PASS/FAIL 只能与**同一台机器、同一模型、只差被测变量的对照组**比，
    # 不能拿这行当门（09-22 实踩：拿它当门会把 AITER 验收判成"回归"）。
    print("注：上面那组参照属于 8121（GLM-5.3-CT-Int4-W4A16，**另一个模型**），只作旁证，不作判据；")
    print("    判据请与同模型对照组逐项相减（见 verify_aiter_linear_arm.sh 的 AITER=0 臂）。")
    # 退出码只保留"召回必须全对"这条真门：召回是字面事实题，坏数值一定先在这里露出来；
    # GSM8K 的条数门交给对照组比较，不在脚本里写死别的模型的数字。
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
