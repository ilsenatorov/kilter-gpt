import math

import pytest

from kiltergpt.utils import KilterPolice, str_to_bool


def test_str_to_bool():
    assert str_to_bool("True") is True
    assert str_to_bool("true") is True
    assert str_to_bool("t") is True
    assert str_to_bool("T") is True
    assert str_to_bool("1") is True
    assert str_to_bool("yes") is True
    assert str_to_bool("Yes") is True
    assert str_to_bool("y") is True
    assert str_to_bool("Y") is True

    assert str_to_bool("False") is False
    assert str_to_bool("false") is False
    assert str_to_bool("f") is False
    assert str_to_bool("F") is False
    assert str_to_bool("0") is False
    assert str_to_bool("no") is False
    assert str_to_bool("No") is False
    assert str_to_bool("n") is False
    assert str_to_bool("N") is False

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
    assert not kilter_police.check("")  # empty climb
    assert not kilter_police.check("p1r12")  # only one hold
