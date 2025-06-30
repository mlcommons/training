from functools import partial

import lightning.pytorch as pl

from ldm.util import instantiate_from_config


class ComposableDataModule(pl.LightningDataModule):

    def __init__(self, train=None, validation=None, test=None, predict=None):
        super().__init__()
        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = partial(self._gen_dataloader, mode="train")
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._gen_dataloader, mode="validation")
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._gen_dataloader, mode="test")
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = partial(self._gen_dataloader, mode="predict")

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

    def _gen_dataloader(self, mode):
        return self.datasets[mode]
