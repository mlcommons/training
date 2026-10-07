"""Gin-driven env-var bootstrap.

Some env vars must be set *before* certain modules import (e.g. Triton's
`@triton.autotune` decorator reads `TRITON_FULL_AUTOTUNE` at module load
time, well before `gin.parse_config_file` runs in the default ordering).

`apply_env_bootstrap()` is `@gin.configurable`, so the gin file becomes the
canonical source of truth. `train_ranker.py` parses gin with
`skip_unknown=True` early in `_main_func`, calls this function to push the
bindings into `os.environ`, then does the heavy imports.
"""

import logging
import os
from typing import Optional

import gin

logger: logging.Logger = logging.getLogger(__name__)


def _bind_bool(name: str, value: Optional[bool]) -> None:
    """Push a gin-bound boolean into the environment; a pre-set env var wins.

    Same env-wins rule as the binding written out below, factored out because
    each additional kernel flag needs it identically.
    """
    if name in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set %s=%s (overrides gin binding)",
            name,
            os.environ[name],
        )
    elif value is not None:
        os.environ[name] = "1" if value else "0"
        logger.info("env bootstrap: %s=%s", name, os.environ[name])


@gin.configurable
def apply_env_bootstrap(
    TRITON_FULL_AUTOTUNE: Optional[bool] = None,
    TRITON_HSTU_LAST_LAYER_TARGETS_ONLY: Optional[bool] = None,
    TRITON_HSTU_TARGETS_ONLY_EVAL: Optional[bool] = None,
) -> None:
    # A pre-set environment variable wins over the gin binding. The pinned
    # triton configs are MI350X-specific, so a different GPU arch (e.g. B200
    # sm_100) sets TRITON_FULL_AUTOTUNE=1 in the launcher environment to
    # re-enable the full autotune search WITHOUT editing this (AMD-default)
    # gin file. Cross-cluster launchers thus stay config-as-code via env.
    if "TRITON_FULL_AUTOTUNE" in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set TRITON_FULL_AUTOTUNE=%s (overrides gin binding)",
            os.environ["TRITON_FULL_AUTOTUNE"],
        )
    elif TRITON_FULL_AUTOTUNE is not None:
        os.environ["TRITON_FULL_AUTOTUNE"] = "1" if TRITON_FULL_AUTOTUNE else "0"
        logger.info("env bootstrap: TRITON_FULL_AUTOTUNE=%s", os.environ["TRITON_FULL_AUTOTUNE"])

    # Read at Triton/module import time, which is why they belong here rather
    # than in a config object the model reads at construction.
    _bind_bool(
        "TRITON_HSTU_LAST_LAYER_TARGETS_ONLY", TRITON_HSTU_LAST_LAYER_TARGETS_ONLY
    )
    _bind_bool("TRITON_HSTU_TARGETS_ONLY_EVAL", TRITON_HSTU_TARGETS_ONLY_EVAL)
