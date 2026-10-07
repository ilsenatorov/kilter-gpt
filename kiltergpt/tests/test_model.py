from argparse import Namespace

import pytest
import torch

from kiltergpt.data.tokenizer import Tokenizer
from kiltergpt.models import GPTModel, GradePredictor
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


def test_cfg_generate(sample_config):
    tokenizer = Tokenizer()
    model = GPTModel(Namespace(**vars(sample_config), p_uncond=0.1), tokenizer).eval()
    generated = model.generate_from_string("p1234r12", 40, "7a", cfg_scale=2.0)
    assert KilterPolice(tokenizer, (1, 2), (1, 2), n_total_holds=(4, 28)).check(generated)
    # null grade replaces the grade entirely
    x, angle = torch.tensor([[1, 100, 6]]), torch.tensor([[40 / 70]])
    uncond = torch.tensor([True])
    torch.testing.assert_close(
        model(x, angle, torch.tensor([[0.2]]), uncond), model(x, angle, torch.tensor([[0.8]]), uncond)
    )
    # checkpoints trained without grade dropout can't guide
    with pytest.raises(ValueError):
        GPTModel(sample_config, tokenizer).generate_from_string("p1234r12", 40, "7a", cfg_scale=2.0)


def test_grade_predictor():
    config = Namespace(n_embed=8, n_head=2, n_layer=1, dropout=0.0, lr=1e-3, wd=0.0, total_steps=10)
    model = GradePredictor(config).eval()
    climb, longer = "p1100r12p1200r13p1300r14", "p1100r12p1200r13p1250r13p1300r14p1350r15"
    alone = model.predict([climb], [40])
    batched = model.predict([climb, longer], [40, 40])
    shuffled = model.predict(["p1300r14p1100r12p1200r13"], [40])
    torch.testing.assert_close(alone, batched[:1])  # padding is masked
    torch.testing.assert_close(alone, shuffled)  # hold order doesn't matter
    police = KilterPolice(Tokenizer(), (1, 2), (1, 2), n_total_holds=(2, 28))
    assert model.grade_mae([climb, longer], [(40, 20.0), (40, 25.0)], police) >= 0


def test_generate_ignores_train_mode(sample_config):
    model = GPTModel(Namespace(**{**vars(sample_config), "dropout": 0.5}, p_uncond=0.5), Tokenizer())
    torch.manual_seed(0)
    in_train_mode = model.generate_from_string("p1234r12", 40, "7a", cfg_scale=2.0)
    assert model.training  # mode restored
    torch.manual_seed(0)
    assert model.eval().generate_from_string("p1234r12", 40, "7a", cfg_scale=2.0) == in_train_mode
