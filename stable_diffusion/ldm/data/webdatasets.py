from torch.utils.data import default_collate
from torchvision import transforms

import webdataset as wds

from ldm.util import instantiate_from_config
from ldm.data.utils import instantiate_transforms_from_config, identity, keys_filter

def build_dataloader(
        urls,
        batch_size,
        shuffle=-1,
        partial=False,
        metadata_filters=None,
        keep_only_keys=None,
        image_transforms=None,
        txt_transforms=None,
        num_workers=1,
        cache_size=-1,
        cache_dir=None,
        persistent_workers=True):
    # TODO(ahmadki): WebDataset supports a "PipeLine" format which is more convenient than
    # the "fluid" format used here. But that one results in an error (TypeError: 'FilterFunction' object is not iterable)
    # which I haven't been able to debug yet.

    image_transforms = transforms.Compose([instantiate_transforms_from_config(t) for t in image_transforms]) if image_transforms is not None else identity
    txt_transforms = transforms.Compose([instantiate_from_config(t) for t in txt_transforms]) if txt_transforms is not None else identity

    dataset = wds.WebDataset(urls=urls, resampled=True, cache_size=cache_size, cache_dir=cache_dir)

    for filter in metadata_filters or []:
        dataset = dataset.select(instantiate_from_config(filter))

    dataset = dataset.shuffle(size=shuffle).decode("pil")

    if keep_only_keys:
        dataset = dataset.map(keys_filter(keep_only_keys))

    if image_transforms or txt_transforms:
        dataset = dataset.map_dict(jpg=image_transforms, txt=txt_transforms)

    dataset = dataset.batched(batch_size, partial=partial, collation_fn=default_collate)
    return wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
