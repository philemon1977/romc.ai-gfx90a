# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 vision variant.

Thin multimodal wrapper around the text-only ``DeepseekV41LLMForCausalLM``:

- ``vision`` ViT + ``aligner`` produce per-image embeddings for the IMAGE
  positions; three learned vectors (``image_start`` / ``image_newline`` /
  ``image_end``) fill the delimiter positions. Every image-span position
  carries ``image_token_id`` (129264) in ``input_ids``; the per-position
  roles come from the processor's ``types`` tensor (see
  ``common/mm_preprocess.py``, and ``common/vision.py`` for the tower).
- Merged embeddings enter the text model via ``inputs_embeds``, i.e. before
  its hyper-connection stream expansion. Raw ``input_ids`` still flow into
  the model so the MoE router can apply ``bias_vl`` to image tokens
  (``requires_raw_input_tokens``).
"""

from collections.abc import Iterable
from typing import Annotated

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
    run_dp_sharded_vision_tower,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from ..common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from .model import (
    DeepseekV41LLMForCausalLM,
    _linear_scale_param_name,
    _make_deepseek_v4_weights_mapper,
)


class DeepseekV4VLImagePixelInputs(TensorSchema):
    """ViT patch inputs for one batched set of images.

    Dimensions:
        - np: Total ViT patches across images (sum of n_vit_h * n_vit_w)
        - c: Number of image channels (3)
        - p: ViT patch size
        - ni: Number of images
        - ns: Total image-span positions (sum of n_llm_h * (n_llm_w + 1) + 2)
    """

    patches: Annotated[
        torch.Tensor, TensorShape("np", 3, "p", "p", dynamic_dims={"np"})
    ]
    # [n_vit_h, n_vit_w] per image
    vit_grid: Annotated[torch.Tensor, TensorShape("ni", 2)]
    # [n_llm_h, n_llm_w] per image
    llm_grid: Annotated[torch.Tensor, TensorShape("ni", 2)]
    types: Annotated[torch.Tensor, TensorShape("ns", dynamic_dims={"ns"})]


def _make_deepseek_v4_vl_weights_mapper(
    expert_dtype: str, linear_scale_name: str
) -> WeightsMapper:
    """Text-checkpoint mapping rules re-rooted under ``language_model.``."""
    base = _make_deepseek_v4_weights_mapper(expert_dtype, linear_scale_name)
    orig_to_new_prefix: dict[str, str | None] = {
        "layers.": "language_model.model.layers.",
        "embed.": "language_model.model.embed.",
        "norm.": "language_model.model.norm.",
        "hc_head": "language_model.model.hc_head",
        "mtp.": "language_model.model.mtp.",
    }
    return WeightsMapper(
        orig_to_new_prefix=orig_to_new_prefix,
        orig_to_new_regex=base.orig_to_new_regex,
        orig_to_new_suffix={
            "head.weight": "language_model.lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            # The MTP/DSpark draft heads are not supported for the vision
            # variant; drop their weights.
            "mtp.": None,
        },
    )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class DeepseekV41ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3):
    """Multimodal entry point for DeepSeek-V4.1 checkpoints with a vision tower.

    ``SupportsEagle3`` (aux hidden-state plumbing for MTP/DSpark drafters)
    delegates through ``language_model`` via the protocol defaults.
    """

    supports_encoder_tp_data = True

    # The MoE router needs raw token ids to detect image-span tokens
    # (all carrying image_token_id, see common/mm_preprocess.py) and apply
    # bias_vl.
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        # The tower is always built; _mark_tower_model stubs it out
        # (StageMissingLayer, weights skipped) when the image limit is 0.
        with self._mark_tower_model(vllm_config, {"image"}):
            self.use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
            self.vision = DeepseekV4ViT(config)
            self.aligner = DeepseekV4Aligner(config)
            self.image_start = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_end = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_newline = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.vision.to(dtype=model_config.dtype)
            self.aligner.to(dtype=model_config.dtype)

        with self._mark_language_model(vllm_config):
            self.language_model = DeepseekV41LLMForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        # The outer mapper (see load_weights) fully resolves HF names into
        # this wrapper's namespace before AutoWeightsLoader strips the
        # "language_model." prefix and delegates to the child's load_weights,
        # so the child's own mapper must be a no-op. Its suffix rules are not
        # idempotent (e.g. "lm_head.weight".endswith("head.weight") would
        # re-fire "head.weight" -> "lm_head.weight").
        self.language_model.hf_to_vllm_mapper = WeightsMapper()
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> DeepseekV4VLImagePixelInputs | None:
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        return DeepseekV4VLImagePixelInputs(
            patches=patches,
            vit_grid=kwargs.pop("vit_grid"),
            llm_grid=kwargs.pop("llm_grid"),
            types=kwargs.pop("types"),
            resolve_bindings={"p": self.config.vision_patch_size},
        )

    def _encode_image(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
    ) -> torch.Tensor:
        # Aligner rows in reading order, one per IMAGE slot.
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def _build_image_span(
        self, image_embeds: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """Full image span: aligner rows at IMAGE slots, the learned
        delimiter vectors at IMAGE_START/IMAGE_NEW_LINE/IMAGE_END."""
        types = types.to(image_embeds.device)
        span = image_embeds.new_empty(types.numel(), image_embeds.shape[-1])
        dtype = image_embeds.dtype
        span[types == IMAGE_START] = self.image_start.to(dtype)
        span[types == IMAGE_END] = self.image_end.to(dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)
        span[types == IMAGE] = image_embeds
        return span

    def _process_image_input(
        self,
        image_input: DeepseekV4VLImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        patches = image_input.patches.to(self.aligner.w1.weight.dtype)
        vit_grid = image_input.vit_grid.tolist()

        image_embeds_list: list[torch.Tensor]
        if self.use_data_parallel and get_tensor_model_parallel_world_size() > 1:
            # Data-parallel ViT: shard images across TP ranks and all-gather
            # the per-image embeddings (weights are replicated on every rank).
            image_embeds_list = run_dp_sharded_vision_tower(
                self.vision, self.aligner, patches, vit_grid
            )
        else:
            image_embeds_list = []
            vit_offset = 0
            for n_vit_h, n_vit_w in vit_grid:
                n_vit = n_vit_h * n_vit_w
                image_embeds_list.append(
                    self._encode_image(
                        patches[vit_offset : vit_offset + n_vit], n_vit_h, n_vit_w
                    )
                )
                vit_offset += n_vit

        embeds: list[torch.Tensor] = []
        span_offset = 0
        for image_embeds, (n_llm_h, n_llm_w) in zip(
            image_embeds_list, image_input.llm_grid.tolist(), strict=True
        ):
            span_len = n_llm_h * (n_llm_w + 1) + 2
            embeds.append(
                self._build_image_span(
                    image_embeds,
                    image_input.types[span_offset : span_offset + span_len],
                )
            )
            span_offset += span_len
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    @staticmethod
    def get_model_state_cls():
        from ..nvidia.model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        return self.language_model.token_lookback_depth

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer for the MTP/DSpark draft model."""
        return self.language_model.get_mtp_target_hidden_states()

    # [gfx90a-host patch] Streaming order-preserving weight load.
    # Upstream `sorted(self.hf_to_vllm_mapper.apply(weights))` materializes
    # EVERY tensor of the checkpoint on CPU *per TP rank* just to order names
    # (8 x 489 GiB anonymous on this 251 GiB host -> swap storm, load never
    # finishes).  And DeepSeek-V4.1 engram tables are single 94 GiB tensors
    # whose param loader narrows to this rank's rows *after* the whole tensor
    # is read (8 x 94 GiB more).  This patch:
    #   pass 1: read only safetensors HEADERS (names), never tensor bytes;
    #   pass 2: stream tensors one at a time in mapped-name order (same
    #           contiguity guarantee the child loader relies on);
    #   engram.embed weight/scale: read this rank's ROW SLICE via get_slice
    #           and copy straight into the param (mirrors
    #           _engram_head_shard_weight_loader incl. e8m0->uint8 view).
    # The upstream `weights` generator is deliberately NOT iterated: an
    # unstarted generator means no I/O was performed.
    _ENGRAM_BIG_RE = __import__("re").compile(
        r"^(.*\.engram\.embed_tokens)\.(weight|weight_scale_inv)$")

    @staticmethod
    def _sf_keys(path):
        from safetensors import safe_open as _so
        with _so(path, framework="pt") as f:
            return [k for k in f.keys() if k != "__metadata__"]

    def _iter_ckpt_names(self, model_path: str):
        """(shard_file, tensor_name, nbytes) for every tensor, header-only."""
        import json as _json
        import os
        wmap = _json.load(
            open(os.path.join(model_path, "model.safetensors.index.json"))
        )["weight_map"]
        out = []
        for f in sorted(set(wmap.values())):
            try:
                with open(os.path.join(model_path, f), "rb") as raw:
                    hn = int.from_bytes(raw.read(8), "little")
                    h = _json.loads(raw.read(hn))
                for k, v in h.items():
                    if k != "__metadata__":
                        out.append((f, k, v["data_offsets"][1] - v["data_offsets"][0]))
            except Exception:
                for k in _sf_keys(os.path.join(model_path, f)):
                    out.append((f, k, 0))
        return out

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        import os as _os
        from safetensors import safe_open as _safe_open
        from vllm.config import get_current_vllm_config

        model_path = get_current_vllm_config().model_config.model
        pairs = list(self._iter_ckpt_names(model_path))  # names only, cheap
        # NB: apply() DROPS entries whose mapping is None (VL discards mtp.*) —
        # map per-name so the kept list stays aligned with source files.
        mapped_triples = []
        for f, k, nb in pairs:
            kept = list(self.hf_to_vllm_mapper.apply([(k, None)]))
            if kept:
                mapped_triples.append((kept[0][0], f, k, nb))
        mapped_triples.sort(key=lambda t: t[0])

        loader = AutoWeightsLoader(self)
        engram_loaded: set[str] = set()

        # Handle cache: one open per shard, reused across its tensors.
        # (per-tensor safe_open() re-parses the multi-MB JSON header — 140k
        # times per rank dominated CPU and starved the disk queue.)
        #
        # [v14] GPU-direct loading.  The swap-death that made v11b/v12/v13 each
        # take 60+ min was NOT the disk: CPU get_tensor copies stacked
        # ~49 GiB of anonymous memory per rank (8×49 > 251 GiB host) -> kernel
        # swapped worker heap -> 20 ms-class page faults -> measured 122 MiB/s.
        # Opening with device="cuda" puts each tensor straight on the device
        # (safetensors stages via small pinned buffers), so CPU anonymous
        # pressure disappears and only reclaimable page cache remains.
        # Engram row-slices KEEP a CPU handle: the param pool has no room for
        # ~12 GiB GPU temporaries (params already occupy 61.64 GiB/rank).
        _handles: dict = {}
        _handles_cpu: dict = {}
        _dev = f"cuda:{torch.cuda.current_device()}"

        def _h(f):
            sf = _handles.get(f)
            if sf is None:
                sf = _safe_open(_os.path.join(model_path, f), framework="pt",
                                device=_dev)
                _handles[f] = sf
            return sf

        def _hc(f):
            sf = _handles_cpu.get(f)
            if sf is None:
                sf = _safe_open(_os.path.join(model_path, f), framework="pt")
                _handles_cpu[f] = sf
            return sf

        GPU_DIRECT_MAX = 64 << 20   # 64 MiB; larger tensors need CPU staging
        ENGROM_CHUNK_GIB = 2.0      # engram row-slice chunk (per-rank CPU peak)

        def _stream():
            for mname, f, orig, nb in mapped_triples:
                m = self._ENGRAM_BIG_RE.match(mname)
                if m is not None:
                    module_path, leaf = m.group(1), m.group(2)
                    try:
                        module = self.get_submodule(module_path)
                    except AttributeError:
                        module = None  # PP rank that does not own it
                    if module is None:
                        continue
                    param = module.get_parameter(leaf)
                    part_rows = param.shape[0]
                    start = int(getattr(param, "engram_vocab_start", 0))
                    sl = _hc(f).get_slice(orig)
                    total_rows = sl.get_shape()[0]
                    row_bytes = max(1, nb // max(1, total_rows))
                    chunk = max(1, int(ENGROM_CHUNK_GIB * (1 << 30)) // row_bytes)
                    for r0 in range(start, start + part_rows, chunk):
                        r1 = min(r0 + chunk, start + part_rows)
                        seg = sl[r0:r1]
                        if seg.dtype == torch.float8_e8m0fnu:
                            seg = seg.view(torch.uint8)
                        elif seg.dtype == torch.float8_e4m3fn and param.dtype == torch.uint8:
                            seg = seg.view(torch.uint8)
                        param.data[r0 - start:r1 - start].copy_(seg)
                        del seg
                    engram_loaded.add(mname)
                    continue
                if nb <= GPU_DIRECT_MAX:
                    yield mname, _h(f).get_tensor(orig)
                else:
                    yield mname, _hc(f).get_tensor(orig)

        try:
            loaded_params = loader.load_weights(_stream())
        finally:
            for sf in list(_handles.values()) + list(_handles_cpu.values()):
                try:
                    sf.__exit__(None, None, None)
                except Exception:
                    pass
            _handles.clear()
            _handles_cpu.clear()
        loaded_params |= engram_loaded
        # The child's load_weights already ran its post-load finalization.
        self._weights_finalized = True
        return loaded_params

    def process_weights_after_loading(self) -> None:
        # Model-level post-load hook (called by the loader after any load
        # format). Under DummyModelLoader the child's load_weights — and
        # hence its finalize step — is bypassed, so run it here instead.
        if getattr(self, "_weights_finalized", False):
            return
        self.language_model.process_weights_after_loading()
