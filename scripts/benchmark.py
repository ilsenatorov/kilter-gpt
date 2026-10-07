import argparse
import json

import lightning.pytorch as L

from kiltergpt.data import KilterDataModule
from kiltergpt.models import GPTModel

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+", help="Local checkpoints to compare on the same post-2024 holdout")
parser.add_argument("--data_dir", default="data/processed-broad-2026")
parser.add_argument("--grade_predictor", default=None, help="GradePredictor checkpoint for grade_mae")
parser.add_argument(
    "--cfg_scales", type=float, nargs="+", default=None, help="Override the checkpoint's guidance scales"
)
parser.add_argument("--limit_test_batches", type=int, default=None, help="For a quick benchmark smoke test")
args = parser.parse_args()

for checkpoint in args.checkpoints:
    dm = KilterDataModule(data_dir=args.data_dir, test_file="test-post-2024.csv")
    trainer = L.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        **({"limit_test_batches": args.limit_test_batches} if args.limit_test_batches else {}),
    )
    model = GPTModel.load_checkpoint(checkpoint)
    model.config.grade_predictor = args.grade_predictor
    if args.cfg_scales:
        model.config.cfg_scales = args.cfg_scales
    scores = trainer.test(model, datamodule=dm, verbose=False)[0]
    print(json.dumps({"checkpoint": checkpoint, "scores": scores}, sort_keys=True))
