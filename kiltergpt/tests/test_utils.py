import pytest
import torch

from kiltergpt.data.datasets import climb_signature
from kiltergpt.data.tokenizer import Tokenizer
from kiltergpt.utils import KilterPolice, Plotter, WarmupCosineSchedule, str_to_bool
from kiltergpt.utils.metrics import generation_metrics


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
        Tokenizer(),
        n_start_holds=(1, 2),
        n_finish_holds=(1, 2),
        n_total_holds=(2, 999),
    )  # p1200 to p1300 are allowed, 1-2 start holds, 1-2 finish holds, 2+ total holds


def test_kilter_police_allowed_things(kilter_police, sane_climb):
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
def test_kilter_police_disallowed_things(kilter_police, sane_climb, broken_addition):
    assert not kilter_police.check(sane_climb + broken_addition)
    assert not kilter_police.check(broken_addition + sane_climb)


def test_kilter_police_too_few(kilter_police):
    assert not kilter_police.check("")  # empty climb
    assert not kilter_police.check("p1200r12")  # only one hold


def test_generation_metrics(kilter_police, sane_climb):
    reordered = "p1203r15p1202r14p1201r13p1200r12"
    different = "p1200r12p1201r13p1202r14p1204r15"
    scores = generation_metrics([reordered, different, ""], {climb_signature(sane_climb)}, kilter_police)
    assert scores["valid_rate"] == 2 / 3
    assert scores["novel_rate"] == 2 / 3
    assert scores["nearest_sampled_distance"] == pytest.approx((0 + 0.4 + 1) / 3)
    assert scores["mean_holds"] == pytest.approx(8 / 3)
    assert scores["mean_feet"] == pytest.approx(2 / 3)
    assert scores["no_feet_rate"] == pytest.approx(1 / 3)


def test_plotter():
    plotter = Plotter()
    climb = "p1234r12p1235r13p1236r14p1237r15"
    normal_plot = plotter.plot_climb(climb)
    assert normal_plot is not None
    matplotlib_plot = plotter.plot_climb(climb, return_fig=True, highlight="p1234")
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


def test_c2st_auc_and_hand_gap():
    from kiltergpt.utils.metrics import c2st_auc, max_hand_gap

    assert max_hand_gap("p1073r12p1387r14p1080r15") > 0.7  # bottom row to near the top, the foot doesn't count
    low = [f"p{1073 + i}r12p{1110 + i}r13p{1150 + i}r14" for i in range(60)]
    high = [f"p{1073 + i}r12p{1300 + i}r13p{1340 + i}r14" for i in range(60)]
    assert c2st_auc(low, high) > 0.95
    assert c2st_auc(low + low, low + low) == 0.5  # duplicated pairs are dropped, identical sets can't be told apart
