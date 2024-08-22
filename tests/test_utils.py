import math

import pytest
import torch

from kiltergpt.utils import KilterPolice, Plotter, WarmupCosineSchedule, str_to_bool


def test_str_to_bool():
    for i in ["True", "true", "t", "T", "1", "yes", "Yes", "y", "Y"]:
        assert str_to_bool(i) is True

    for i in ["False", "false", "f", "F", "0", "no", "No", "n", "N"]:
        assert str_to_bool(i) is False

    with pytest.raises(ValueError):
        str_to_bool("invalid")
        str_to_bool("2")
        str_to_bool("nope")


@pytest.fixture
def sane_climb():
    return "p1200r12p1201r13p1202r14p1203r15"


@pytest.fixture
def kilter_police():
    return KilterPolice(
        allowed_holds=set([i for i in range(1200, 1300)]),
        n_start_holds=(1, 2),
        n_finish_holds=(1, 2),
        n_total_holds=(2, math.inf),
    )  # p1200 to p1300 are allowed, 1-2 start holds, 1-2 finish holds, 2+ total holds


def kilter_police_allowed_things(kilter_police, sane_climb):
    assert kilter_police.check(sane_climb)


@pytest.mark.parametrize(
    "broken_addition",
    [
        "p1250r16",  # broken color
        "p4000r15",  # broken hold
        "p1250r12p1251r12p1252r12",  # too many start holds
        "p1250r14p1251r14p1252r14",  # too many finish holds
    ],
)
def kilter_police_disallowed_things(kilter_police, sane_climb, broken_addition):
    assert not kilter_police.check(sane_climb + broken_addition)
    assert not kilter_police.check(broken_addition + sane_climb)


def kilter_police_too_few(kilter_police):
    assert not kilter_police.check("")  # empty climb
    assert not kilter_police.check("p1200r12")  # only one hold


def test_plotter():
    plotter = Plotter()
    climb = "p1234r12p1235r13p1236r14p1237r15"
    normal_plot = plotter.plot_climb(climb)
    assert normal_plot is not None
    matplotlib_plot = plotter.plot_climb(climb, return_fig=True)
    assert matplotlib_plot is not None


def test_scheduler():
    base_lr = 1e-3
    start_lr_coeff = 0.1
    end_lr_coeff = 0.01
    optim = torch.optim.Adam([torch.nn.Parameter(torch.randn(1))], lr=base_lr)
    scheduler = WarmupCosineSchedule(
        optim, warmup_steps=10, total_steps=100, start_lr_coeff=start_lr_coeff, end_lr_coeff=end_lr_coeff
    )
    lrs = []
    for _ in range(150):
        lrs.append(scheduler.get_last_lr()[0])
        optim.step()
        scheduler.step()
    for lr in lrs[:10]:
        assert base_lr * start_lr_coeff <= lr <= base_lr
    for lr in lrs[10:100]:
        assert base_lr * end_lr_coeff <= lr <= base_lr
    for lr in lrs[100:]:
        assert lr == base_lr * end_lr_coeff
