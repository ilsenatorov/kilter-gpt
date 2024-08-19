from argparse import ArgumentParser

import torch
import wandb
from fastapi import FastAPI

from kiltergpt.models.gpt import GPTModel

parser = ArgumentParser()
parser.add_argument("model_name", type=str, help="Name of the model from wandb, something like 'model-hhh777xxx:best'")

args = parser.parse_args()

app = FastAPI()

model = GPTModel.load_from_wandb(args.model_name).to("cpu")
model.eval()


@app.get("/generate_climb")
def generate_climb(frames: str, angle: int, difficulty: str, temperature: float = 0.2, p: float = 1.0):
    with torch.no_grad():
        result = model.generate_from_string(frames, angle, difficulty, temperature, p)
    return {"climb": result[0]}
