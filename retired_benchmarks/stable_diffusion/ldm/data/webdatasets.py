from torch.utils.data import default_collate
from torchvision import transforms

import webdataset as wds

from ldm.util import instantiate_from_config
from ldm.data.utils import instantiate_transforms_from_config, identity, keys_filter

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def build_dataloader(
        urls,
        batch_size,
        shuffle=-1,
        partial=False,
        decode=None,
        metadata_filters=None,
        keep_only_keys=None,
        transformations=None,
        num_workers=1,
        cache_size=-1,
        cache_dir=None,
        persistent_workers=True):
    # TODO(ahmadki): WebDataset supports a "PipeLine" format which is more convenient than the "fluid" format used here
    # But fluid format results in an error (TypeError: 'FilterFunction' object is not iterable)
    # which I haven't been able to debug yet.
    dataset = wds.WebDataset(urls=urls, resampled=True, cache_size=cache_size, cache_dir=cache_dir)

    # Filter samples based on metadata
    for filter in metadata_filters or []:
        dataset = dataset.select(instantiate_from_config(filter))

    # shuffle
    dataset = dataset.shuffle(size=shuffle)

    # decode
    if isinstance(decode, str):
        dataset = dataset.decode(decode)
    else:
        dataset = dataset.decode()

    # Filter keys
    if keep_only_keys:
        dataset = dataset.map(keys_filter(keep_only_keys))

    # Apply transformations
    if transformations is not None:
        transformations_dict = {k: transforms.Compose([instantiate_transforms_from_config(t) for t in transformations[k]]) for k in transformations.keys()}
        dataset = dataset.map_dict(**transformations_dict)

    dataset = dataset.batched(batch_size, partial=partial, collation_fn=default_collate)
    return wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
