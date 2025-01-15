# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import enum
import os
import pathlib

from huggingface_hub import snapshot_download
from nemo.collections.llm.gpt.model.mixtral import HFMixtralImporter


class Model(enum.Enum):
    MIXTRAL_8x7B_BASE = "mistralai/Mixtral-8x7B-v0.1"
    MIXTRAL_8x22B_BASE = "mistralai/Mixtral-8x22B-v0.1"

    def __str__(self):
        return self.value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_id",
        type=str,
        default=Model.MIXTRAL_8x7B_BASE.value,
        choices=list(Model) + ["path"],
    )

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    assert (
        args.hf_token != None and args.checkpoint_id in list(Model)
    ), "You must provide HF Token as either command line argument or HF_TOKEN env variable"

    if args.checkpoint_id in list(Model):
        snapshot_download(
            repo_id=str(args.checkpoint_id),
            repo_type="model",
            local_dir=args.output_dir / "hf",
            token=args.hf_token,
        )
        importer = HFMixtralImporter(args.output_dir / "hf")
    else:
        importer = HFMixtralImporter(args.checkpoint_id)
    importer.apply(args.output_dir / "nemo")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
