#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import sys
from pathlib import Path

# Primus and Megatron paths (set by run_and_time.sh or here as fallback)
PRIMUS_PATH = os.getenv("PRIMUS_PATH", "/workspace/deps/Primus")
MEGATRON_PATH = os.path.join(PRIMUS_PATH, "third_party/Megatron-LM")

if PRIMUS_PATH not in sys.path:
    sys.path.insert(0, PRIMUS_PATH)
if MEGATRON_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_PATH)

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import load_primus_config, add_pretrain_parser
from primus_mllog import MLPerfMegatronPretrainTrainer

import argparse

def setup_environment(data_path: str = None):
    """Setup HuggingFace home and other environment variables."""
    if data_path and "HF_HOME" not in os.environ:
        hf_home = os.path.join(data_path, "huggingface")
        os.environ["HF_HOME"] = hf_home
        print(f"[MLPerf Train] HF_HOME={hf_home}")


def load_config(config_path: str, overrides: list = None) -> PrimusConfig:
    """
    Load and parse the experiment YAML configuration.
    
    The config file (e.g., gpt_oss_20B-pretrain.yaml) defines:
    - Model architecture (hidden size, num layers, attention heads, etc.)
    - Training hyperparameters (batch size, learning rate, etc.)
    - Data paths and tokenizer settings
    - Parallelism settings (TP, PP, EP)
    """
    # Create args namespace for Primus config loader
    parser = argparse.ArgumentParser()
    add_pretrain_parser(parser)
    
    args = parser.parse_args([
        '--config', config_path,
        '--data_path', os.getenv('DATA_PATH', '/data'),
    ])
    
    primus_cfg, unknown_overrides = load_primus_config(args, overrides or [])
    
    print(f"[MLPerf Train] Loaded config from: {config_path}")
    print(f"[MLPerf Train] Framework: {primus_cfg.get_module_config('pre_trainer').framework}")
    
    return primus_cfg, unknown_overrides


def create_trainer(primus_cfg: PrimusConfig, extra_args: list = None) -> MLPerfMegatronPretrainTrainer:
    """
    Create the MLPerf-enabled Megatron trainer.
    
    The trainer handles:
    - Model creation (GPT architecture with MoE)
    - Optimizer setup (Adam with configurable betas)
    - Learning rate scheduling (warmup + cosine decay)
    - Distributed training coordination
    - MLPerf logging and metrics
    """
    # Get distributed training configuration from environment
    # These are set by torchrun when launching distributed training
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    trainer = MLPerfMegatronPretrainTrainer(
        module_name="pre_trainer",
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
        extra_args=extra_args,
    )
    return trainer

def main():
    config_path = os.environ.get("EXP", "/workspace/code/conf/gpt_oss_20B-pretrain.yaml")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    setup_environment(data_path=os.getenv('DATA_PATH', '/data'))
    primus_cfg, extra_args = load_config(config_path)
    
    trainer = create_trainer(primus_cfg, extra_args)
    trainer.init()
    trainer.run()
    
if __name__ == "__main__":
    main()
