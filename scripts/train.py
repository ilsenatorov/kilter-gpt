from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from src.kiltergpt.train import train
from src.kiltergpt.utils import str_to_bool

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# dataset params
parser.add_argument("--subset", type=float, default=1.0, help="Fraction of datasets to take")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--max_num_tokens", type=int, default=None, help="Max number of tokens, overrides batch size")
parser.add_argument("--smooth_labels", type=str_to_bool, default=False, help="Smooth labels")
# training params
parser.add_argument("--only_train", type=str_to_bool, default=False, help="Skip the testing part")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--lr", type=float, default=6e-4, help="Max learning rate")
parser.add_argument("--wd", type=float, default=1e-1, help="Weight decay")
parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")
# model params
parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
parser.add_argument("--n_layer", type=int, default=8, help="Number of transformer layers")
parser.add_argument("--n_embed", type=int, default=512, help="Embedding dimension")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
parser.add_argument("--bias", type=str_to_bool, default=False, help="Use bias in attention layers")
parser.add_argument("--context_len", type=int, default=64, help="Context length")
config = parser.parse_args()

train(config)
