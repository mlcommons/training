"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

try:
    from google.cloud import storage

    HAS_IMPORT_GOOGLE_CLOUD_SDK_EXCEPTION = None
except ImportError as e:
    HAS_IMPORT_GOOGLE_CLOUD_SDK_EXCEPTION = e


def parse_gcs_bucket_and_blob_name(gcs_path):
    splits = gcs_path.replace("gs://", "").split("/", 1)
    bucket = splits[0]
    blob_name = "" if len(splits) == 1 else splits[1]
    return bucket, blob_name


def get_blob(gcs_path):
    bucket, blob_name = parse_gcs_bucket_and_blob_name(gcs_path)
    assert blob_name, f"{blob_name=} should be a valid name"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(blob_name)
    return blob


def get_file(path, mode):
    if path.startswith("gs://"):
        if HAS_IMPORT_GOOGLE_CLOUD_SDK_EXCEPTION:
            raise HAS_IMPORT_GOOGLE_CLOUD_SDK_EXCEPTION
        return get_blob(path).open(mode)
    else:
        file_dir = os.path.dirname(path)
        os.makedirs(file_dir, exist_ok=True)
        return open(path, mode)
