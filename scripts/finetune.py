import argparse

import lightning.pytorch as L

from kiltergpt.data import KilterDataModule
from kiltergpt.models.gpt import GPTModel
from kiltergpt.utils import str_to_bool

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, default="data/finetune", help="Data directory")
parser.add_argument("--prompts", type=str, nargs="+", default=["p1233r12p1228r12"], help="Prompts to test for")
parser.add_argument("--max_num_tokens", type=int, default=2048, help="Maximum number of tokens")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
parser.add_argument("--smooth_labels", type=str_to_bool, default=True, help="Smooth labels")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs")
parser.add_argument("--checkpoint_path", type=str, default="ilsenatorov/model-registry/kiltergpt:best")
config = parser.parse_args()

dm = KilterDataModule(
    data_dir=config.data_dir,
    max_num_tokens=config.max_num_tokens,
    num_workers=config.num_workers,
    smooth_labels=config.smooth_labels,
)
dm.setup()
model = GPTModel.load_from_wandb(config.checkpoint_path)
model.config.only_train = False
model.config.lr = 1e-5
model.config.total_steps = len(dm.train_dataloader()) * config.epochs
trainer = L.Trainer(
    max_epochs=config.epochs,
    gradient_clip_val=5.0,
    logger=L.loggers.WandbLogger(project="kilter-gpt-finetune", config=config, log_model=True),
)


def check_n_footholds(model, prompts: list[str], n_iter: int = 100):
    count = {}
    for prompt in prompts:
        count[prompt] = 0
        for _ in range(n_iter):
            frames = model.generate_from_string(prompt, 40, "7a")
            if "r15" in frames:
                count[prompt] += 1
    return count


trainer.logger.log_metrics({f"footholds/{k}": v for k, v in check_n_footholds(model, config.prompts).items()})
trainer.test(model, dm)
trainer.fit(model, dm)
trainer.logger.log_metrics({f"footholds/{k}": v for k, v in check_n_footholds(model, config.prompts).items()})
trainer.test(model, dm)
