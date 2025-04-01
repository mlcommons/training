# Lint as: python3
"""Script to load layer(s) of the LLM checkpoint using TensorStore.
More details about TensorStore, please visit 
https://github.com/google/tensorstore .
"""

import argparse
import tensorstore as ts

parser = argparse.ArgumentParser(description='Checkpoint loading for LLM.')
parser.add_argument(
    '--input_path',
    type=str,
    default='',
    help='Input directory for layer(s) of the saved checkpoint.')
args = parser.parse_args()

if __name__ == '__main__':
  input_path = args.input_path
  spec = {'driver': 'zarr', 'kvstore': {}}
  spec['kvstore'] = {
      'driver': 'file',
      'path': input_path,
  }
  t = ts.open(ts.Spec(spec), open=True).result()
  t_v = t.read().result()

  print("path = ", input_path,
        ", type = ", type(t_v),
        ", shape = ", t_v.shape)
