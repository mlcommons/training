# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

import logging
import os
from typing import Dict, Optional, Set, Tuple

import torch
from generative_recommenders.common import fx_unwrap_optional_tensor, HammerModule
from generative_recommenders.modules.positional_encoder import HSTUPositionalEncoder
from generative_recommenders.modules.postprocessors import (
    L2NormPostprocessor,
    OutputPostprocessor,
)
from generative_recommenders.modules.preprocessors import InputPreprocessor
from generative_recommenders.modules.stu import STU
from generative_recommenders.ops.jagged_tensors import split_2D_jagged
from torch.profiler import record_function

logger: logging.Logger = logging.getLogger(__name__)
torch.fx.wrap("len")
_HSTU_LAST_LAYER_TARGETS_ONLY: bool = (
    os.environ.get("TRITON_HSTU_LAST_LAYER_TARGETS_ONLY", "0") == "1"
)
# Whether eval takes the targets-only path too. Separately switchable because
# eval numerics feed the convergence metric the RCP sweep is compared against,
# so an A/B has to be able to move eval without moving training.
_HSTU_TARGETS_ONLY_EVAL: bool = (
    os.environ.get("TRITON_HSTU_TARGETS_ONLY_EVAL", "1") == "1"
)
_TARGETS_ONLY_CHECKED: bool = False
_TARGETS_ONLY_SKIPS: Set[str] = set()


def _log_targets_only_skip_once(reason: str) -> None:
    if reason not in _TARGETS_ONLY_SKIPS:
        _TARGETS_ONLY_SKIPS.add(reason)
        logger.warning(
            "[hstu] last layer: falling back to the full path for this batch "
            f"({reason}); targets-only needs exactly one target per sequence"
        )

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@torch.fx.wrap
def default_seq_payload(
    seq_payloads: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if seq_payloads is None:
        return {}
    else:
        return torch.jit._unwrap_optional(seq_payloads)


class HSTUTransducer(HammerModule):
    def __init__(
        self,
        stu_module: STU,
        input_preprocessor: InputPreprocessor,
        output_postprocessor: Optional[OutputPostprocessor] = None,
        input_dropout_ratio: float = 0.0,
        positional_encoder: Optional[HSTUPositionalEncoder] = None,
        is_inference: bool = True,
        return_full_embeddings: bool = False,
        listwise: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_module = stu_module
        self._input_preprocessor: InputPreprocessor = input_preprocessor
        self._output_postprocessor: OutputPostprocessor = (
            output_postprocessor
            if output_postprocessor is not None
            else L2NormPostprocessor(is_inference=is_inference)
        )
        assert self._is_inference == self._input_preprocessor._is_inference, (
            f"input_preprocessor must have the same mode; self: {self._is_inference} vs input_preprocessor {self._input_preprocessor._is_inference}"
        )
        self._positional_encoder: Optional[HSTUPositionalEncoder] = positional_encoder
        self._input_dropout_ratio: float = input_dropout_ratio
        self._return_full_embeddings: bool = return_full_embeddings
        self._listwise_training: bool = listwise and self.is_train
        # Whether the last forward actually took the targets-only path. Read by
        # DlrmHSTU to price the FLOPs counter, which would otherwise bill the
        # last layer at full sequence length. Per-batch, because the gate can
        # legitimately decline (eval, or not one target per sequence).
        self._last_targets_only: bool = False

        for name, m in self.named_modules():
            if "_stu_module" in name:
                continue
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, torch.nn.LayerNorm):
                if m.weight.dim() >= 2:
                    torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None and m.bias.dim() >= 2:
                    torch.nn.init.xavier_normal_(m.bias)

    def _preprocess(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        seq_payloads = default_seq_payload(seq_payloads)

        with record_function("hstu_input_preprocessor"):
            (
                output_max_seq_len,
                output_total_uih_len,
                output_total_targets,
                output_seq_lengths,
                output_seq_offsets,
                output_seq_timestamps,
                output_seq_embeddings,
                output_num_targets,
                output_seq_payloads,
            ) = self._input_preprocessor(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
                seq_lengths=seq_lengths,
                seq_timestamps=seq_timestamps,
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                seq_payloads=seq_payloads,
            )

        with record_function("hstu_positional_encoder"):
            if self._positional_encoder is not None:
                output_seq_embeddings = self._positional_encoder(
                    max_seq_len=output_max_seq_len,
                    seq_lengths=output_seq_lengths,
                    seq_offsets=output_seq_offsets,
                    seq_timestamps=output_seq_timestamps,
                    seq_embeddings=output_seq_embeddings,
                    num_targets=(
                        None if self._listwise_training else output_num_targets
                    ),
                )

        output_seq_embeddings = torch.nn.functional.dropout(
            output_seq_embeddings,
            p=self._input_dropout_ratio,
            training=self.training,
        )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            output_seq_payloads,
        )

    @torch.jit.unused
    def _targets_only_unsupported_reason(self) -> Optional[str]:
        """Why targets-only can never run in this configuration, or None.

        Only conditions fixed for the whole run belong here. Anything that can
        differ between two batches of the same run is decided per call instead.
        """
        if self._is_inference:
            return "transducer is in inference mode"
        if self._return_full_embeddings:
            return "return_full_embeddings=True consumes every row, not just candidates"
        if self._listwise_training:
            return "listwise training does not pass num_targets to attention"
        if self._input_preprocessor.interleave_targets():
            return "preprocessor interleaves targets, so they are not a suffix"
        return self._stu_module.targets_only_unsupported_reason()

    @torch.jit.unused
    def _resolve_targets_only(
        self, max_targets: int, total_targets: int, batch_size: int
    ) -> bool:
        """Decide the last-layer targets-only path, and fail loudly if asked for
        something this configuration can never do.

        The knob defaults off, so reaching here means it was explicitly turned
        on. A config that cannot satisfy it is then a mistake worth raising on:
        a silent fallback would read as "the optimization simply didn't help",
        and nobody re-checks a knob that appears to be working. A batch that is
        not exactly one target per sequence is not a mistake, so that only falls
        back, logged once per distinct reason so a rare shape cannot hide. A
        device without the separated-RNG dropout path falls back the same way:
        that is not a mistake either, since no config change can fix it.

        Eval takes the same path as training. The two are exact in math -- with
        one target per sequence the standard path's ``split_2D_jagged`` selects
        precisely the row targets-only computes, and both feed it through the
        same output postprocessor -- so the only difference is reduction order.
        Dropout is keyed off ``self.training`` inside the layer, not off this
        decision, so it stays off in eval either way.
        """
        global _TARGETS_ONLY_CHECKED
        if not _TARGETS_ONLY_CHECKED:
            _TARGETS_ONLY_CHECKED = True
            reason = self._targets_only_unsupported_reason()
            if reason is not None:
                raise RuntimeError(
                    "TRITON_HSTU_LAST_LAYER_TARGETS_ONLY=1 was requested but "
                    f"this configuration cannot support it: {reason}. Set "
                    "TRITON_HSTU_LAST_LAYER_TARGETS_ONLY=0 (or gin "
                    "apply_env_bootstrap.TRITON_HSTU_LAST_LAYER_TARGETS_ONLY"
                    "=False) to run the standard last layer instead."
                )
            logger.info(
                "[hstu] last layer: targets-only (one candidate row per sequence)"
            )
        # Hardware, unlike the conditions above, is not something the caller can
        # go fix, so this declines instead of raising. Training needs the
        # candidate row to draw the same dropout mask the full-length pass would
        # have drawn, and only the separated-RNG path can index a mask that way.
        # Eval is unaffected: dropout is off, so no mask has to match.
        from generative_recommenders.ops.triton.triton_hstu_linear import (
            supports_indexed_output_dropout,
        )

        if self.training and not supports_indexed_output_dropout():
            _log_targets_only_skip_once(
                "device has no separated-RNG dropout mask path, so the compact "
                "row cannot reproduce the full-length dropout stream"
            )
            return False
        if not self.training and not _HSTU_TARGETS_ONLY_EVAL:
            _log_targets_only_skip_once("eval (TRITON_HSTU_TARGETS_ONLY_EVAL=0)")
            return False
        if max_targets != 1 or total_targets != batch_size:
            _log_targets_only_skip_once(
                f"max_targets={max_targets} total_targets={total_targets} "
                f"batch_size={batch_size}"
            )
            return False
        return True

    def _hstu_compute(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        targets_only: bool,
    ) -> torch.Tensor:
        with record_function("hstu"):
            if targets_only:
                seq_embeddings = self._stu_module.forward_targets_only(
                    max_seq_len=max_seq_len,
                    x=seq_embeddings,
                    x_lengths=seq_lengths,
                    x_offsets=seq_offsets,
                    num_targets=num_targets,
                )
            else:
                seq_embeddings = self._stu_module(
                    max_seq_len=max_seq_len,
                    x=seq_embeddings,
                    x_lengths=seq_lengths,
                    x_offsets=seq_offsets,
                    num_targets=(None if self._listwise_training else num_targets),
                )
        return seq_embeddings

    def _postprocess(
        self,
        max_seq_len: int,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        targets_only: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        with record_function("hstu_output_postprocessor"):
            if targets_only:
                seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                    seq_lengths
                )
                candidate_rows = seq_offsets[1:] - 1
                candidate_timestamps = torch.index_select(
                    seq_timestamps, 0, candidate_rows
                )
                candidate_embeddings = self._output_postprocessor(
                    seq_embeddings=seq_embeddings,
                    seq_timestamps=candidate_timestamps,
                    seq_payloads=seq_payloads,
                )
                return None, candidate_embeddings
            if self._return_full_embeddings:
                seq_embeddings = self._output_postprocessor(
                    seq_embeddings=seq_embeddings,
                    seq_timestamps=seq_timestamps,
                    seq_payloads=seq_payloads,
                )
            uih_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                seq_lengths - num_targets
            )
            candidates_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_targets
            )
            _, candidate_embeddings = split_2D_jagged(
                values=seq_embeddings,
                max_seq_len=max_seq_len,
                total_len_left=total_uih_len,
                total_len_right=total_targets,
                max_len_left=max_uih_len,
                max_len_right=max_targets,
                offsets_left=uih_offsets,
                offsets_right=candidates_offsets,
                kernel=self.hammer_kernel(),
            )
            interleave_targets: bool = self._input_preprocessor.interleave_targets()
            if interleave_targets:
                candidate_embeddings = candidate_embeddings.view(
                    -1, 2, candidate_embeddings.size(-1)
                )[:, 0, :]
            if not self._return_full_embeddings:
                _, candidate_timestamps = split_2D_jagged(
                    values=seq_timestamps.unsqueeze(-1),
                    max_seq_len=max_seq_len,
                    total_len_left=total_uih_len,
                    total_len_right=total_targets,
                    max_len_left=max_uih_len,
                    max_len_right=max_targets,
                    offsets_left=uih_offsets,
                    offsets_right=candidates_offsets,
                    kernel=self.hammer_kernel(),
                )
                candidate_timestamps = candidate_timestamps.squeeze(-1)
                if interleave_targets:
                    candidate_timestamps = candidate_timestamps.view(-1, 2)[:, 0]
                candidate_embeddings = self._output_postprocessor(
                    seq_embeddings=candidate_embeddings,
                    seq_timestamps=candidate_timestamps,
                    seq_payloads=seq_payloads,
                )

            return (
                seq_embeddings if self._return_full_embeddings else None,
                candidate_embeddings,
            )

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        orig_dtype = seq_embeddings.dtype
        if not self._is_inference:
            seq_embeddings = seq_embeddings.to(self._training_dtype)

        (
            max_seq_len,
            total_uih_len,
            total_targets,
            seq_lengths,
            seq_offsets,
            seq_timestamps,
            seq_embeddings,
            num_targets,
            seq_payloads,
        ) = self._preprocess(
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
        )

        targets_only = False
        if _HSTU_LAST_LAYER_TARGETS_ONLY and not torch.jit.is_scripting():
            targets_only = self._resolve_targets_only(
                max_targets=max_targets,
                total_targets=total_targets,
                batch_size=seq_lengths.size(0),
            )
        if not torch.jit.is_scripting():
            self._last_targets_only = targets_only
        encoded_embeddings = self._hstu_compute(
            max_seq_len=max_seq_len,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            targets_only=targets_only,
        )

        encoded_embeddings, encoded_candidate_embeddings = self._postprocess(
            max_seq_len=max_seq_len,
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_embeddings=encoded_embeddings,
            seq_timestamps=seq_timestamps,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
            targets_only=targets_only,
        )

        if not self._is_inference:
            encoded_candidate_embeddings = encoded_candidate_embeddings.to(orig_dtype)
            if self._return_full_embeddings:
                encoded_embeddings = fx_unwrap_optional_tensor(encoded_embeddings).to(
                    orig_dtype
                )
        return (
            encoded_candidate_embeddings,
            encoded_embeddings,
        )
