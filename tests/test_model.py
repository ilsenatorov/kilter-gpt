from argparse import Namespace

import pytest

from kiltergpt.data.tokenizer import Tokenizer
from kiltergpt.models.gpt import GPTModel


@pytest.fixture
def sample_config():
    tokenizer = Tokenizer()
    return Namespace(
        label_smoothing=True,
        batch_size=2,
        epochs=3,
        lr=6e-4,
        wd=1e-1,
        n_head=4,
        n_layer=4,
        n_embed=128,
        context_len=64,
        dropout=0.2,
        bias=False,
        vocab_size=tokenizer.vocab_size,
        total_steps=300,
    )


def test_forward_pass(sample_config):
    tokenizer = Tokenizer()
    model = GPTModel(sample_config, tokenizer)
    sample_input = tokenizer.encode(
        "p1234r12p1345r13p1423r14p1243r15",
        40,
        "7a",
        pad=sample_config.context_len,
        shuffle=True,
    )
    sample_batch = sample_input.unsqueeze(0).repeat(2, 1)
    output = model(sample_batch)
    assert output.shape == (2, 64, sample_config.vocab_size)


def test_generate(sample_config):
    tokenizer = Tokenizer()
    model = GPTModel(sample_config, tokenizer)
    sample_prompt = "p1234r12"
    generated = model.generate_from_string(sample_prompt, 40, "7a")
    print(generated)
    # TODO write better tests
