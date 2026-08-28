#!/usr/bin/env python3

# pyre-strict

import copy
import unittest
from unittest import mock

import fbgemm_gpu  # noqa: F401
import torch
from generative_recommenders.common import (
    gpu_unavailable,
    HammerKernel,
    set_dev_mode,
)
from generative_recommenders.modules.stu import STULayer, STULayerConfig, STUStack


class STUTargetsOnlyTest(unittest.TestCase):
    def _make_layer(
        self,
        dropout_ratio: float = 0.3,
        is_inference: bool = False,
        **overrides: object,
    ) -> STULayer:
        config = dict(
            embedding_dim=64,
            num_heads=2,
            hidden_dim=32,
            attention_dim=32,
            output_dropout_ratio=dropout_ratio,
            causal=True,
            target_aware=True,
            use_group_norm=False,
            recompute_normed_x=True,
            recompute_uvqk=True,
            recompute_y=True,
            sort_by_length=True,
            contextual_seq_len=0,
        )
        config.update(overrides)
        layer = STULayer(
            config=STULayerConfig(**config),  # pyre-ignore [6]
            is_inference=is_inference,
        ).to(device="cuda", dtype=torch.bfloat16)
        layer.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
        return layer

    @unittest.skipIf(*gpu_unavailable)
    def test_output_gradients_and_rng_match_full_path(self) -> None:
        set_dev_mode(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.tensor([5, 9, 17, 31], device="cuda", dtype=torch.int64)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        num_targets = torch.ones_like(lengths)
        candidate_rows = offsets[1:] - 1
        total_rows = int(offsets[-1].item())

        reference = self._make_layer()
        candidate = copy.deepcopy(reference)
        torch.manual_seed(2026)
        x_reference = torch.randn(
            total_rows,
            64,
            device="cuda",
            dtype=torch.bfloat16,
        ).requires_grad_()
        x_candidate = x_reference.detach().clone().requires_grad_()

        torch.manual_seed(12345)
        full_output = reference(
            x=x_reference,
            x_lengths=lengths,
            x_offsets=offsets,
            max_seq_len=int(lengths.max().item()),
            num_targets=num_targets,
        )
        expected = torch.index_select(full_output, 0, candidate_rows)
        expected_post_forward_rng = torch.rand(32, device="cuda")

        torch.manual_seed(12345)
        actual = candidate.forward_targets_only(
            x=x_candidate,
            x_lengths=lengths,
            x_offsets=offsets,
            max_seq_len=int(lengths.max().item()),
            num_targets=num_targets,
        )
        actual_post_forward_rng = torch.rand(32, device="cuda")

        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(
            actual_post_forward_rng,
            expected_post_forward_rng,
            atol=0.0,
            rtol=0.0,
        )

        torch.manual_seed(67890)
        dout = torch.randn_like(expected)
        expected.backward(dout)
        actual.backward(dout)
        torch.testing.assert_close(
            x_candidate.grad,
            x_reference.grad,
            atol=5e-2,
            rtol=5e-2,
        )
        for (reference_name, reference_parameter), (
            candidate_name,
            candidate_parameter,
        ) in zip(reference.named_parameters(), candidate.named_parameters()):
            self.assertEqual(reference_name, candidate_name)
            self.assertIsNotNone(reference_parameter.grad)
            self.assertIsNotNone(candidate_parameter.grad)
            torch.testing.assert_close(
                candidate_parameter.grad,
                reference_parameter.grad,
                atol=5e-2,
                rtol=5e-2,
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_activation_contract_rejects_unsupported_layers(self) -> None:
        set_dev_mode(True)
        supported = self._make_layer(dropout_ratio=0.0)
        self.assertTrue(supported.supports_targets_only())

        grouped = STULayer(
            config=STULayerConfig(
                embedding_dim=64,
                num_heads=2,
                hidden_dim=32,
                attention_dim=32,
                use_group_norm=True,
            ),
            is_inference=False,
        ).to(device="cuda", dtype=torch.bfloat16)
        grouped.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
        self.assertFalse(grouped.supports_targets_only())

        inference = self._make_layer(dropout_ratio=0.0, is_inference=True)
        self.assertFalse(inference.supports_targets_only())

    @unittest.skipIf(*gpu_unavailable)
    def test_rejection_names_the_offending_condition(self) -> None:
        """A rejection has to say which condition closed the gate.

        The caller raises with this string, so "unsupported" alone would leave
        someone to bisect nine conditions by hand.
        """
        set_dev_mode(True)
        self.assertIsNone(
            self._make_layer(dropout_ratio=0.0).targets_only_unsupported_reason()
        )

        for kwargs, expected in (
            ({"causal": False}, "causal"),
            ({"target_aware": False}, "target-aware"),
            ({"max_attn_len": 128}, "max_attn_len"),
            ({"contextual_seq_len": 4}, "contextual_seq_len"),
            ({"recompute_uvqk": False}, "recompute_uvqk"),
        ):
            with self.subTest(**kwargs):
                reason = self._make_layer(
                    dropout_ratio=0.0, **kwargs
                ).targets_only_unsupported_reason()
                self.assertIsNotNone(reason)
                self.assertIn(expected, reason)

        stack = STUStack([self._make_layer(dropout_ratio=0.0, causal=False)])
        self.assertIn("causal", stack.targets_only_unsupported_reason() or "")
        self.assertIn(
            "no STU layers", STUStack([]).targets_only_unsupported_reason() or ""
        )

    def test_delta_backward_does_not_specialize_on_max_seq_len(self) -> None:
        """max_seq_len must stay a runtime scalar in the delta backward.

        Production derives it per batch as max_uih_len + num_candidates, so as a
        tl.constexpr it keys the JIT cache on a value that changes nearly every
        step. A cold process then recompiles the kernel per batch -- measured at
        ~100 ms each, 32 compiles for 32 distinct values -- and because the
        compile blocks the launch queue it shows up as GPU idle, not host time.
        A warm cache hides all of it, so no throughput or numerics test in this
        suite would notice the regression.
        """
        from generative_recommenders.ops.triton.triton_hstu_attention import (
            _hstu_attn_delta_bwd,
        )

        constexpr_params = {
            param.name for param in _hstu_attn_delta_bwd.params if param.is_constexpr
        }
        self.assertNotIn("MAX_SEQ_LEN", constexpr_params)
        # Only shape-independent tuning constants may be baked in: anything
        # derived from the batch would reintroduce the per-step recompile.
        self.assertTrue(
            all(name.startswith("BLOCK_") for name in constexpr_params),
            f"unexpected compile-time constant(s): {sorted(constexpr_params)}",
        )

    def test_eval_takes_the_targets_only_path(self) -> None:
        """Eval must run the same shrunken last layer that training does.

        The rewrite is exact for any batch that is one target per sequence, and
        the holdout batches are, so gating it on ``self.training`` only made
        eval pay a full-length last layer -- worth 5.6% of eval time on an
        otherwise identical pair. Guarded because the gate is one ``and`` away
        from quietly excluding eval again, and no numerics test would notice.
        """
        from generative_recommenders.modules import hstu_transducer as ht

        class _Stub:
            training = False

            def _targets_only_unsupported_reason(self):
                return None

        stub = _Stub()
        with mock.patch.object(ht, "_TARGETS_ONLY_CHECKED", True):
            for knob in (True, False):
                with mock.patch.object(ht, "_HSTU_TARGETS_ONLY_EVAL", knob):
                    self.assertIs(
                        ht.HSTUTransducer._resolve_targets_only(
                            stub, max_targets=1, total_targets=4, batch_size=4
                        ),
                        knob,
                    )
            # A shape the rewrite cannot express still falls back in eval.
            with mock.patch.object(ht, "_HSTU_TARGETS_ONLY_EVAL", True):
                self.assertIs(
                    ht.HSTUTransducer._resolve_targets_only(
                        stub, max_targets=2, total_targets=8, batch_size=4
                    ),
                    False,
                )

    def test_device_without_indexed_dropout_declines_rather_than_raises(self) -> None:
        """A device with no separated-RNG mask path must fall back, not raise.

        Training needs the candidate row to draw the same dropout mask the
        full-length pass would have drawn, and only the separated-RNG path can
        index a mask that way; below sm_100 (and pre-MI350) it does not exist.
        Every other blocked condition here is a config the caller can go fix, so
        raising is the right answer for those. Hardware is not, and raising on it
        would turn a default-on knob into a hard failure on those GPUs. Eval is
        exempt because dropout is off, so no mask has to match.

        This is the one gate condition that cannot be reached on the hardware
        this suite runs on, so it is asserted by substitution or not at all.
        """
        from generative_recommenders.modules import hstu_transducer as ht
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        class _Training:
            training = True

            def _targets_only_unsupported_reason(self):
                return None

        class _Eval(_Training):
            training = False

        with mock.patch.object(ht, "_TARGETS_ONLY_CHECKED", True), mock.patch.object(
            ht, "_HSTU_TARGETS_ONLY_EVAL", True
        ):
            for supported, expected_in_training in ((False, False), (True, True)):
                with mock.patch.object(
                    thl, "supports_indexed_output_dropout", lambda: supported
                ):
                    self.assertIs(
                        ht.HSTUTransducer._resolve_targets_only(
                            _Training(), max_targets=1, total_targets=4, batch_size=4
                        ),
                        expected_in_training,
                    )
                    # Eval takes the path either way.
                    self.assertIs(
                        ht.HSTUTransducer._resolve_targets_only(
                            _Eval(), max_targets=1, total_targets=4, batch_size=4
                        ),
                        True,
                    )


if __name__ == "__main__":
    unittest.main()
