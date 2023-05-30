import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from transformers import AutoTokenizer

try:
    import lightning.pytorch as pl  # latest
except ModuleNotFoundError:
    import pytorch_lightning as pl  # older

from nan import NANTokenizer
from .hf_utils import get_processed_dataset


class LitData(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path="allenai/lila",
        name="GSM8k_structured",
        split=None,
        filter_kwargs=dict(col_filter=dict(output_answer="is_number")),
        transform_kwargs=dict(col_transform=dict(output_answer="to_number")),
        resplit_kwargs=dict(test_split="validation", test_size=0.1, seed=42),
        tokenizer_model="bert-base-uncased",
        tokenizer_wrapper="NANTokenizer",
        max_seq_length=None,
        text_col="input",
        text_pair_col=None,
        out_cols="output_answer",
        prune_cols=True,
        dataloader_kwargs=dict(batch_size=64, shuffle=False, num_workers=2),
        **kwargs,
    ):

        super().__init__()

        self.dataset_path = dataset_path
        self.name = name
        self.split = split
        self.filter_kwargs = deepcopy(filter_kwargs)
        self.transform_kwargs = deepcopy(transform_kwargs)
        self.resplit_kwargs = deepcopy(resplit_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)
        if tokenizer_wrapper == "NANTokenizer":
            self.tokenizer = NANTokenizer(tokenizer=self.tokenizer)

        self.max_seq_length = max_seq_length

        self.text_col = text_col
        self.text_pair_col = text_pair_col
        self.out_cols = [out_cols] if isinstance(out_cols, str) else out_cols
        self.prune_cols = prune_cols

        self.dataset = None  # created while setting up
        self.setup()

        self.dataloader_kwargs = deepcopy(dataloader_kwargs)

    def setup(self, stage="fit"):
        self.dataset = get_processed_dataset(
            dataset_path=self.dataset_path,
            name=self.name,
            split=self.split,
            filter_kwargs=self.filter_kwargs,
            transform_kwargs=self.transform_kwargs,
            resplit_kwargs=self.resplit_kwargs,
        )

        prune_cols = self.prune_cols
        # If not a given list then infer before expanding cols in dataset
        if prune_cols is True:
            prune_cols = set(self.dataset["train"].column_names) - set(self.out_cols)

        self.dataset = self.dataset.map(self.convert_to_features, batched=True)
        if prune_cols:
            self.dataset = self.dataset.remove_columns(prune_cols)

        self.dataset.set_format(type="torch")

        # uncomment below for debugging with smaller data
        # for split in self.dataset.keys():
        #     self.dataset[split] = self.dataset[split].select(
        #         range(min(10, len(self.dataset[split])))
        #     )

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], **self.dataloader_kwargs)

    def val_dataloader(self):
        if self.dataset.get("validation") and len(self.dataset["validation"]) > 0:
            return DataLoader(self.dataset["validation"], **self.dataloader_kwargs)

    def test_dataloader(self):
        if self.dataset.get("test") and len(self.dataset["test"]) > 0:
            return DataLoader(self.dataset["test"], **self.dataloader_kwargs)

    def convert_to_features(self, ex, indices=None):
        text = ex[self.text_col]
        text_pair = ex[self.text_pair_col] if self.text_pair_col else None

        features = self.tokenizer(
            text,
            text_pair,
            max_length=self.max_seq_length,
            padding=True if self.max_seq_length else "max_length",
            truncation=True,
            return_tensors=None,
        )

        return features
