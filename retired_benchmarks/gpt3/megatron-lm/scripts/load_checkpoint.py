import argparse
import megatron
import os
import torch

parser = argparse.ArgumentParser(description='Load Megatron-LM checkpoint')
parser.add_argument(
    '--input_path',
    type=str,
    required=True,
    help='Input directory for checkpoint.')
parser.add_argument(
    '--tensor-model-parallel-size',
    type=int,
    default=8)
parser.add_argument(
    '--pipeline-model-parallel-size',
    type=int,
    default=8)
args = parser.parse_args()

if __name__ == '__main__':
    input_path = args.input_path
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    data = []    

    for tp in range(tensor_model_parallel_size):
        for pp in range(pipeline_model_parallel_size):
            file_path = os.path.join(input_path, 
                        'mp_rank_{:02d}_{:03d}'.format(tp,pp),
                        'model_optim_rng.pt')
            data.append(torch.load(file_path, map_location='cpu'))
