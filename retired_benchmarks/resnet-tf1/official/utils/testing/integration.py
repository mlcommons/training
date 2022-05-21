# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Helper code to run complete models from within python.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import shutil
import sys
import tempfile


def run_synthetic(main, tmp_root, extra_flags=None, synth=True, max_train=1):
  """Performs a minimal run of a model.

    This function is intended to test for syntax errors throughout a model. A
  very limited run is performed using synthetic data.

  Args:
    main: The primary function used to exercise a code path. Generally this
      function is "<MODULE>.main(argv)".
    tmp_root: Root path for the temp directory created by the test class.
    extra_flags: Additional flags passed by the caller of this function.
    synth: Use synthetic data.
    max_train: Maximum number of allowed training steps.
  """

  extra_flags = [] if extra_flags is None else extra_flags

  model_dir = tempfile.mkdtemp(dir=tmp_root)

  args = [sys.argv[0], "--model_dir", model_dir, "--train_epochs", "1",
          "--epochs_between_evals", "1"] + extra_flags

  if synth:
    args.append("--use_synthetic_data")

  if max_train is not None:
    args.extend(["--max_train_steps", str(max_train)])

  try:
    main(args)
  finally:
    if os.path.exists(model_dir):
      shutil.rmtree(model_dir)
