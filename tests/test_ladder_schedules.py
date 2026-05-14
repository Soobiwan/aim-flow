from aim_flow.ladder import parse_aggregation_steps, select_reference_condition_index


def test_progressive_reference_starts_at_base_and_ends_at_final():
    assert select_reference_condition_index(0, 24, 4, "progressive") == 0
    assert select_reference_condition_index(23, 24, 4, "progressive") == 3


def test_parse_aggregation_fractions_for_24_steps():
    steps = parse_aggregation_steps(None, [0.125, 0.333, 0.583, 0.833], 24)

    assert steps == {3, 8, 14, 20}


def test_explicit_aggregation_steps_override_fractions():
    steps = parse_aggregation_steps([3, 8, 14, 20], [0.1], 24)

    assert steps == {3, 8, 14, 20}
