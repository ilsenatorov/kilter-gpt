"""Export a GPT checkpoint to ONNX for the Android app and check it against PyTorch.

Inputs: tokens (B, T) int64, angle (B, 1) float (angle / 70), grade (B, 1) float ((grade index + 1) / 39),
for checkpoints trained with grade dropout, uncond (B,) bool, and for checkpoints trained with --popularity,
popularity (B, 1) float (Tokenizer.popularity of the ascent count). Output: logits (B, T + 2, vocab), T + 3 with popularity;
the last position predicts the next token."""

from argparse import ArgumentParser

import numpy as np
import onnxruntime
import torch

from kiltergpt.models import GPTModel

parser = ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("output")
args = parser.parse_args()

model = GPTModel.load_checkpoint(args.checkpoint).model.eval()
guided = model.null_grade is not None
popular = model.popularity_proj is not None
if popular and not guided:
    raise ValueError("Exporting popularity without guidance isn't supported")  # uncond comes first positionally
names = ["tokens", "angle", "grade"] + (["uncond"] if guided else []) + (["popularity"] if popular else [])

# conditional and unconditional rows in one batch, as GPTModel._generate_token does for guidance
tokens = torch.tensor([[1, 323, 7, 100, 5]] * 2)
angle, grade = torch.full((2, 1), 40 / 70), torch.full((2, 1), 22 / 39)
inputs = (tokens, angle, grade) + ((torch.tensor([False, True]),) if guided else ())
inputs += (torch.full((2, 1), 0.5),) if popular else ()

batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", max=model.config.context_len)
torch.onnx.export(
    model,
    inputs,
    args.output,
    input_names=names,
    output_names=["logits"],
    dynamic_shapes=({0: batch, 1: seq},) + ({0: batch},) * (len(names) - 1),
    dynamo=True,
    external_data=False,
)

session = onnxruntime.InferenceSession(args.output)
for length in (1, 5, 62):  # check a few prompt lengths, not just the traced one
    feed = (torch.randint(9, 485, (2, length)), angle, grade) + inputs[3:]
    expected = model(*feed).detach().numpy()
    actual = session.run(None, {n: x.numpy() for n, x in zip(names, feed, strict=True)})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
print(f"Exported {args.output} with inputs {names}; matches PyTorch")
