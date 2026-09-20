#!/usr/bin/env python3
"""QR C2/C3 marker 检查（可独立运行；容器内跑请先 docker cp）。"""
import pathlib, sys
V = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators")
q = V / "quick_all_reduce.py"
c = V / "cuda_communicator.py"
if not q.is_file() or not c.is_file():
    print("NO_FILES"); sys.exit(2)
qt, ct = q.read_text(), c.read_text()
assign_ok = any(l.strip().startswith("supported_archs") for l in qt.splitlines())
gfx_ok = "gfx90" in qt
cond_ok = "if self.world_size > 1 and current_platform.is_rocm():" in ct
print("%s %s %s" % (assign_ok, gfx_ok, cond_ok))
sys.exit(0 if (assign_ok and gfx_ok and cond_ok) else 1)