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
import os
import pathlib
import subprocess
import tempfile

from huggingface_hub import snapshot_download

training_files = [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(768, 1024)]
validation_files = [f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]

file_mapping_train = [
    (f"c4-train.en_{i}.json.gz", f"c4_train.en_{i}") for i in range(6, 8)
]


def download_dataset(
    output_path: pathlib.Path,
    repo_id: str = "allenai/c4",
) -> None:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_path,
        allow_patterns="en/*.json.gz",
    )


def merge_into_consolidated(
    source_directory: pathlib.Path,
    output_directory: pathlib.Path,
):
    def merge_files(output_path: pathlib.Path, input_paths: list[pathlib.Path]):
        with open(output_path, "wb") as output_file:
            for input_path in input_paths:
                with open(input_path, "rb") as input_file:
                    file_content = input_file.read()
                output_file.write(file_content)

    for i in range(8):
        file_chunks = [
            source_directory / training_files[j] for j in range(i * 128, (i + 1) * 128)
        ]
        merge_files(output_directory / f"c4-train.en_{i}.json.gz", file_chunks)

    file_chunks = [source_directory / x for x in validation_files]
    merge_files(output_directory / "c4-validation.en.json.gz", file_chunks)


def run_conversion(
    input_file: pathlib.Path,
    output_file: pathlib.Path,
    tokenizer_path: pathlib.Path,
):
    print(f"Converting {input_file} into {output_file} using {tokenizer_path}")

    output = subprocess.run(
        [
            "python",
            "/opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--tokenizer-library",
            "sentencepiece",
            "--tokenizer-model",
            str(tokenizer_path),
            "--dataset-impl",
            "mmap",
            "--workers",
            "128",
        ],
        capture_output=True,
    )

    print(f"Exited with code={output.returncode}")
    print(f"> Output:\n{str(output.stdout)}")
    print(f"> Error:\n{str(output.stderr)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        required=True,
        help="Path to store output dataset",
    )
    parser.add_argument(
        "--input-tokenizer",
        type=pathlib.Path,
        required=True,
        help="Path for stored tokenizer",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path("/cache")

        os.makedirs(tmpdir_path / "raw", exist_ok=True)
        os.makedirs(tmpdir_path / "merged", exist_ok=True)

        download_dataset(tmpdir_path / "raw")
        merge_into_consolidated(tmpdir_path / "raw", tmpdir_path / "merged")

        for source, target in file_mapping_train:
            run_conversion(
                tmpdir_path / "merged" / source,
                args.output_path / target,
                args.input_tokenizer,
            )
        run_conversion(
            tmpdir_path / "raw" / "en/c4-validation_24567exp.json",
            args.output_path / "c4-validation-small.en",
            args.input_tokenizer,
        )
