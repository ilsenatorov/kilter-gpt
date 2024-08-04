import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from .datasets import KilterGPTDataset
from .tokenizer import Tokenizer


class KilterDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
        context_len: int = 64,
        min_tokens: int = 10,
        label_smoothing: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.context_len = context_len
        self.min_tokens = min_tokens
        self.label_smoothing = label_smoothing

    def setup(self, stage=None):
        df = pd.read_csv("data/processed/all_climbs.csv")
        self.tokenizer = Tokenizer.from_df(df)
        self.train = KilterGPTDataset(
            "data/processed/train.csv",
            self.tokenizer,
            context_len=self.context_len,
            min_tokens=self.min_tokens,
            label_smoothing=self.label_smoothing,
        )

        self.val = KilterGPTDataset(
            "data/processed/val.csv",
            self.tokenizer,
            context_len=self.context_len,
            min_tokens=self.min_tokens,
            label_smoothing=self.label_smoothing,
        )
        self.test = KilterGPTDataset(
            "data/processed/test.csv",
            self.tokenizer,
            context_len=self.context_len,
            min_tokens=self.min_tokens,
            label_smoothing=self.label_smoothing,
        )
        self.test.eval = True
        self.vocab_size = self.tokenizer.vocab_size

    def _get_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True, num_workers=16)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test)
