from aim_flow.ladder import get_active_condition_indices, parse_aggregation_steps, select_reference_condition_index
from aim_flow.sampler import AIMFlowSampler


def test_sparse_steps_only_aggregate_configured_indices():
    steps = parse_aggregation_steps([3, 8, 14, 20], None, 24)

    assert [idx for idx in range(24) if idx in steps] == [3, 8, 14, 20]


def test_non_aggregation_reference_policy_selects_reference():
    reference = select_reference_condition_index(8, 24, 4, "progressive")

    assert AIMFlowSampler._select_non_aggregation_index("reference", reference, 3) == reference


def test_active_policy_around_reference_is_valid_and_clipped():
    assert get_active_condition_indices(0, 24, 4, "around_reference", 0) == [0, 1]
    assert get_active_condition_indices(10, 24, 4, "around_reference", 2) == [1, 2, 3]
    assert get_active_condition_indices(23, 24, 4, "around_reference", 3) == [2, 3]
