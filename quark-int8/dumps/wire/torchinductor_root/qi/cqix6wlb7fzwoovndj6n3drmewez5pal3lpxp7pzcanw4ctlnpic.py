# AOT ID: ['2_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/zw/czwqpfay6po3ykj3lett7byfqhtxk76enhrqoawsutlhlgjikmwm.py
# Topologically Sorted Source Nodes: [gate, gate_1, silu, up, up_1, mul], Original ATen: [aten.slice, aten.clamp, aten.silu, aten.mul, aten.copy_]
# Source node to ATen node mapping:
#   gate => slice_2
#   gate_1 => clamp_max, convert_element_type
#   mul => mul_14
#   silu => add_18, convert_element_type_5, div, exp, neg
#   up => slice_4
#   up_1 => clamp_max_1, clamp_min, convert_element_type_2, convert_element_type_3
# Graph fragment:
#   %arg3_1 : Tensor "bf16[s31, s81][s81, 1]cuda:4" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "bf16[s31, s90][s90, 1]cuda:4" = PlaceHolder[target=buf0]
#   %copy_ : Tensor "bf16[s31, s90][s90, 1]cuda:4" = PlaceHolder[target=copy_]
#   %slice_2 : Tensor "bf16[s31, (s81//2)][s81, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, %floordiv), kwargs = {})
#   %convert_element_type : Tensor "f32[s31, (s81//2)][Max(1, (s81//2)), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_2, torch.float32), kwargs = {})
#   %clamp_max : Tensor "f32[s31, (s81//2)][Max(1, (s81//2)), 1]cuda:4"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%convert_element_type, 10.0), kwargs = {})
#   %neg : Tensor "f32[s31, (s81//2)][Max(1, (s81//2)), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%clamp_max,), kwargs = {})
#   %exp : Tensor "f32[s31, (s81//2)][Max(1, (s81//2)), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_18 : Tensor "f32[s31, (s81//2)][Max(1, (s81//2)), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s31, (s81//2)][Max(1, (s81//2)), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max, %add_18), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[s31, (s81//2)][Max(1, (s81//2)), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_4 : Tensor "bf16[s31, s81 - ((s81//2))][s81, 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, %floordiv, 9223372036854775807), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[s31, s81 - ((s81//2))][Max(1, s81 - ((s81//2))), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_4, torch.float32), kwargs = {})
#   %clamp_min : Tensor "f32[s31, s81 - ((s81//2))][Max(1, s81 - ((s81//2))), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, -10.0), kwargs = {})
#   %clamp_max_1 : Tensor "f32[s31, s81 - ((s81//2))][Max(1, s81 - ((s81//2))), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 10.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s31, s81 - ((s81//2))][Max(1, s81 - ((s81//2))), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_1, torch.bfloat16), kwargs = {})
#   %mul_14 : Tensor "bf16[s31, (s81//2)][Max(1, (s81//2)), 1]cuda:4"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, %convert_element_type_3), kwargs = {})
#   %copy_ : Tensor "bf16[s31, s90][s90, 1]cuda:4"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg5_1, %mul_14), kwargs = {})
#   return %buf0,%buf1
triton_poi_fused_clamp_copy__mul_silu_slice_0 = async_compile.triton('triton_poi_fused_clamp_copy__mul_silu_slice_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=4, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_copy__mul_silu_slice_0', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '9B7B4BF3E513717766BF1B43F7411CBEBB2EE8CF97883A4ED1B6530B3F9CAF4A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 113246208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_copy__mul_silu_slice_0(in_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + ks1*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (x0 + ks1*x1 + (ks1 // 2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 10.0, tl.float32)
    tmp3 = tl.minimum(tmp1, tmp2, tl.PropagateNan.ALL)
    tmp4 = -tmp3
    tmp5 = libdevice.exp(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = (tmp3 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1], -10.0, tl.float32)
    tmp13 = tl.maximum(tmp11, tmp12, tl.PropagateNan.ALL)
    tmp14 = tl.minimum(tmp13, tmp2, tl.PropagateNan.ALL)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        s31 = arg0_1
        s81 = arg1_1
        s32 = arg2_1
        s90 = arg4_1
        assert_size_stride(arg3_1, (s31, s81), (s81, 1))
        assert_size_stride(arg5_1, (s31, s90), (s90, 1))
        with torch.cuda._DeviceGuard(4):
            torch.cuda.set_device(4)
            # Topologically Sorted Source Nodes: [gate, gate_1, silu, up, up_1, mul], Original ATen: [aten.slice, aten.clamp, aten.silu, aten.mul, aten.copy_]
            triton_poi_fused_clamp_copy__mul_silu_slice_0_xnumel = s31*s90
            raw_stream4 = get_raw_stream(4)
            triton_poi_fused_clamp_copy__mul_silu_slice_0.run(arg3_1, arg5_1, s90, s81, triton_poi_fused_clamp_copy__mul_silu_slice_0_xnumel, stream=raw_stream4)
            del arg3_1
            del arg5_1
        return ()

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 49152
    arg1_1 = 576
    arg2_1 = 251658240
    arg3_1 = rand_strided((49152, 576), (576, 1), device='cuda:4', dtype=torch.bfloat16)
    arg4_1 = 288
    arg5_1 = rand_strided((49152, 288), (288, 1), device='cuda:4', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
