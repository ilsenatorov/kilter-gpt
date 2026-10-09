from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from kiltergpt.train import train
from kiltergpt.utils import str_to_bool

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# dataset params
parser.add_argument("--data_dir", type=str, default="data/processed", help="Preprocessed train/val/test directory")
parser.add_argument("--subset", type=float, default=1.0, help="Fraction of datasets to take")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size when max_num_tokens is unset")
parser.add_argument("--max_num_tokens", type=int, default=4096, help="Max number of tokens, overrides batch size")
parser.add_argument("--test_subset", type=float, default=1.0, help="Random fraction of the test set to generate from")
parser.add_argument("--smooth_labels", type=str_to_bool, default=False, help="Smooth labels")
parser.add_argument(
    "--popularity", type=str_to_bool, default=False, help="Condition on ascent count (with p_uncond dropout)"
)
parser.add_argument("--role_first", type=str_to_bool, default=False, help="Tokenize each hold as role then placement")
parser.add_argument("--bottom_up", type=str_to_bool, default=False, help="Order holds bottom to top, not randomly")
# training params
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--only_train", type=str_to_bool, default=False, help="Skip the testing part")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--lr", type=float, default=6e-4, help="Max learning rate")
parser.add_argument("--wd", type=float, default=1e-1, help="Weight decay")
parser.add_argument("--precision", type=str, default="32-true", help="Training precision")
parser.add_argument("--p_uncond", type=float, default=0.1, help="Grade dropout for classifier-free guidance")
parser.add_argument("--cfg_scales", type=float, nargs="+", default=[1.0, 3.0], help="Guidance scales to test")
parser.add_argument("--grade_predictor", type=str, default=None, help="GradePredictor checkpoint for test/grade_mae")
# model params
parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
parser.add_argument("--n_layer", type=int, default=8, help="Number of transformer layers")
parser.add_argument("--n_embed", type=int, default=512, help="Embedding dimension")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
parser.add_argument("--bias", type=str_to_bool, default=False, help="Use bias in attention layers")
parser.add_argument("--hold_positions", type=str_to_bool, default=False, help="Add board positions to hold embeddings")
parser.add_argument(
    "--spatial_attention", type=str_to_bool, default=False, help="Bias attention between holds by board distance"
)
parser.add_argument("--context_len", type=int, default=64, help="Context length")
config = parser.parse_args()

train(config)
