import math

import pytest

from kiltergpt.utils import KilterPolice, Plotter, str_to_bool


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
    return "p1r12p2r13p3r14p4r15"


@pytest.fixture
def kilter_police():
    return KilterPolice(
        allowed_holds=set([i for i in range(1, 10)]),
        n_start_holds=(1, 2),
        n_finish_holds=(1, 2),
        n_total_holds=(2, math.inf),
    )  # p1 to p9 are allowed, 1-2 start holds, 1-2 finish holds, 2+ total holds


def kilter_police_allowed_things(kilter_police, sane_climb):
    assert kilter_police.check(sane_climb)


@pytest.mark.parametrize(
    "broken_addition",
    [
        "p5r16",  # broken color
        "p20r15",  # broken hold
        "p1r12p1r12p1r12",  # too many start holds
        "p1r14p1r14p1r14",  # too many finish holds
    ],
)
def kilter_police_disallowed_things(kilter_police, sane_climb, broken_addition):
    assert not kilter_police.check(sane_climb + broken_addition)
    assert not kilter_police.check(broken_addition + sane_climb)


def kilter_police_too_few(kilter_police):
    assert not kilter_police.check("")  # empty climb
    assert not kilter_police.check("p1r12")  # only one hold


def test_plotter():
    plotter = Plotter()
    normal_plot = plotter.plot_climb("p1234r12")
    assert normal_plot is not None
    matplotlib_plot = plotter.plot_climb("p1234r12", True)
    assert matplotlib_plot is not None
