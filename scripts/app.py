from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from fastapi import FastAPI

from kiltergpt.models.gpt import GPTModel

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", type=str, default="kiltergpt:best", help="Name of the model to load")
parser.add_argument("--repo_name", type=str, default="ilsenatorov/model-registry", help="Where to look for the model")

args = parser.parse_args()

app = FastAPI()

model = GPTModel.load_from_wandb(args.model_name, args.repo_name).to("cpu")
app = model.get_fastapi_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
