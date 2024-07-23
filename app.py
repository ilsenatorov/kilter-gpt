import torch
from fastapi import FastAPI

from src.models.gpt import GPTModel

app = FastAPI()

MODEL_NAME = "model-h7iskggx:v0"

# run = wandb.init(name="model_download")
# artifact = run.use_artifact(f"ilsenatorov/kilter-gpt/{MODEL_NAME}", type="model")
# artifact_dir = artifact.download()
# wandb.finish()

model = GPTModel.load_from_checkpoint(f"artifacts/{MODEL_NAME}/model.ckpt").to("cpu")
model.eval()
for i in range(5):
    print(f"Doing test run {i}/5")
    model.generate_from_string("p1100r12", 40, "7a")


@app.get("/generate_climb")
def generate_climb(frames: str, angle: int, difficulty: str, temperature: float = 0.2, p: float = 1.0):
    with torch.no_grad():
        result = model.generate_from_string(frames, angle, difficulty, temperature, p)
    return {"climb": result[0]}
