from torchvision import transforms
from einops import rearrange

from ldm.util import instantiate_from_config

def instantiate_transforms_from_config(config):
    if config.target in ['torchvision.transforms.RandomResizedCrop', 'torchvision.transforms.Resize']:
        # the isinstance is necessary because instantiate_transforms_from_config might be called multiple times
        # and isinstance(config['params']['interpolation'] already caseted from str to InterpolationMode
        if "interpolation" in config['params'] and isinstance(config['params']['interpolation'], str):
            config.params.interpolation = interpolation_from_string(config['params']['interpolation'])
    return instantiate_from_config(config)

def interpolation_from_string(interpolation):
    interpolation_map = {
        'nearest': transforms.InterpolationMode.NEAREST,
        'bilinear': transforms.InterpolationMode.BILINEAR,
        'bicubic': transforms.InterpolationMode.BICUBIC,
        'box': transforms.InterpolationMode.BOX,
        'hamming': transforms.InterpolationMode.HAMMING,
        'lanczos': transforms.InterpolationMode.LANCZOS,
    }
    return interpolation_map[interpolation]

def rearrange_transform(pattern):
    return transforms.Lambda(lambda x: rearrange(tensor=x, pattern=pattern))

def identity(x):
    return x

def keys_filter(keys):
    def filter_fn(sample):
        return {k: v for k, v in sample.items() if k in keys}
    return filter_fn

def value_filter(key, predicate, value):
    def filter_fn(sample):
        if predicate == "eq":
            return sample[key] == value
        elif predicate == "neq":
            return sample[key] != value
        elif predicate == "gt":
            return sample[key] > value
        elif predicate == "lt":
            return sample[key] < value
        elif predicate == "gte":
            return sample[key] >= value
        elif predicate == "lte":
            return sample[key] <= value
        else:
            raise ValueError(f"Unknown predicate: {predicate}")
    return filter_fn
