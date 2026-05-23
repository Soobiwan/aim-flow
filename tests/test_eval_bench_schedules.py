import pytest

from aim_flow.eval_bench.constants import DEFAULT_SPFC_SCHEDULE_1_INDEXED, SPFC_SCHEDULE_ABLATIONS_1_INDEXED
from aim_flow.eval_bench.schedules import get_schedule_ablation, one_indexed_to_zero_indexed


def test_default_schedule_converts_to_sampler_indices():
    assert one_indexed_to_zero_indexed(DEFAULT_SPFC_SCHEDULE_1_INDEXED, num_steps=24, required_count=16) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        13,
        15,
        17,
        19,
        23,
    ]


def test_all_schedule_ablations_have_sixteen_steps():
    for name, schedule in SPFC_SCHEDULE_ABLATIONS_1_INDEXED.items():
        assert get_schedule_ablation(name) == one_indexed_to_zero_indexed(schedule, required_count=16)


def test_schedule_rejects_out_of_range_and_duplicates():
    with pytest.raises(ValueError, match="outside"):
        one_indexed_to_zero_indexed([0], num_steps=24)
    with pytest.raises(ValueError, match="Duplicate"):
        one_indexed_to_zero_indexed([1, 1], num_steps=24)
