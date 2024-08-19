from argparse import ArgumentParser

import torch
from fastapi import FastAPI

from kiltergpt.models.gpt import GPTModel

parser = ArgumentParser()
parser.add_argument("model_name", type=str, help="Name of the model from wandb, something like 'model-hhh777xxx:best'")

args = parser.parse_args()

app = FastAPI()

model = GPTModel.load_from_wandb(args.model_name).to("cpu")
app = model.get_fastapi_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
