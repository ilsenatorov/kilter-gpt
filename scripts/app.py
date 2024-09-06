from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from fastapi import FastAPI

from kiltergpt.models.gpt import GPTModel

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="ilsenatorov/kilter-gpt-finetune/model-gkuck0fe:v0",
    help="Name of the model to load",
)
args = parser.parse_args()

app = FastAPI()

model = GPTModel.load_from_wandb(args.checkpoint_path).to("cpu")
app = model.get_fastapi_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
