import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.data.datasets import KilterGPTDataset
from src.models.gpt import GPTModel

torch.set_float32_matmul_precision("medium")


class Config:
    batch_size = 2048
    epochs = 1000
    vocab_size = 536
    lr = 6e-4
    wd = 1e-5
    n_embed = 256
    num_blocks = 4
    num_heads = 4
    head_size = n_embed // num_heads
    context_len = 32
    attn_drop_value = 0.2
    multihead_drop_value = 0.2
    ffn_drop_value = 0.2


config_dict = dict(vars(Config))
del [config_dict["__module__"]]
del [config_dict["__dict__"]]
del [config_dict["__weakref__"]]
del [config_dict["__doc__"]]

config = Config()

ds = KilterGPTDataset("kilterboard.csv", context_len=config.context_len)
dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
model = GPTModel(config)

trainer = Trainer(
    devices=-1,
    max_epochs=config.epochs,
    logger=[WandbLogger(project="kilter-gpt")],
    # detect_anomaly=True,
)

trainer.fit(model, dl)
