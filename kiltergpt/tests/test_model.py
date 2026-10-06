from argparse import Namespace

import pytest
from fastapi.testclient import TestClient

from kiltergpt.data.tokenizer import Tokenizer
from kiltergpt.models.gpt import GPTModel
from kiltergpt.utils import KilterPolice


@pytest.fixture
def sample_config():
    tokenizer = Tokenizer()
    return Namespace(
        label_smoothing=True,
        batch_size=2,
        epochs=3,
        lr=6e-4,
        wd=1e-1,
        n_head=2,
        n_layer=2,
        n_embed=4,
        context_len=64,
        dropout=0.2,
        bias=False,
        vocab_size=tokenizer.vocab_size,
        total_steps=300,
    )


def test_generate(sample_config):
    tokenizer = Tokenizer()
    model = GPTModel(sample_config, tokenizer)
    sample_prompt = "p1234r12"
    generated = model.generate_from_string(sample_prompt, 40, "7a")
    assert sample_prompt in generated
    assert KilterPolice(tokenizer, (1, 2), (1, 2), n_total_holds=(4, 28)).check(generated)


@pytest.mark.parametrize("prompt", ["p1234r12p1234r14", "p1234r16", "p1234r12oops", "p1234"])
def test_invalid_prompt(sample_config, prompt):
    model = GPTModel(sample_config, Tokenizer())
    with pytest.raises(ValueError):
        model.generate_from_string(prompt, 40, "7a")


def test_app(sample_config):
    tokenizer = Tokenizer()
    model = GPTModel(sample_config, tokenizer)
    app = model.get_fastapi_app()
    sample_prompt = "p1234r12p1333r14"
    client = TestClient(app)
    response = client.get(
        "/generate",  # Update the URL path to match the endpoint in your FastAPI application
        params={"frames": sample_prompt, "angle": 40, "grade": "7a", "temperature": 0.1, "p": 0.7},
    )
    assert response.status_code == 200
    assert sample_prompt in response.json()["climb"]
    assert (
        client.get("/generate", params={"frames": "p1234r12p1234r14", "angle": 40, "grade": "7a"}).status_code == 422
    )
    assert (
        client.get("/generate", params={"frames": sample_prompt, "angle": 40, "grade": "7a", "p": 0}).status_code
        == 422
    )
